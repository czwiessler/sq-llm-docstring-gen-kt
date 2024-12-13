
from flask import current_app

import db
from db.data import count_all_lowlevel
import db.exceptions
import similarity.metrics
import similarity.utils
from utils.list_utils import chunks
import json

from sqlalchemy import text
from collections import defaultdict

def get_all_metrics():
    """Get name, category, and description for each of the metrics
    in the similarity.similarity_metrics table.

    Returns:
        metrics (dict): {category: [metric_name, description],
                            ...,
                         category_n: [metric_name, description]}
    """
    with db.engine.begin() as connection:
        result = connection.execute("""
            SELECT category
                 , metric
                 , description
              FROM similarity.similarity_metrics
        """)

        metrics = defaultdict(list)
        for category, metric, description in result.fetchall():
            metrics[category].append([metric, description])

        return metrics


def get_similarity_count():
    """Get total number of ids from similarity table.

    Returns:
        num_ids (int): number of ids from similarity.similarity
        table.
    """
    with db.engine.connect() as connection:
        query = text("""
            SELECT count(*)
              FROM similarity.similarity
        """)
        result = connection.execute(query)
        num_ids = result.fetchone()[0]
        if not num_ids:
            raise db.exceptions.NoDataFoundException("Similarity metrics are not computed for any submissions.")
        return num_ids


def get_similarity_ids():
    """Gets all ids from similarity table in an ascending list.

    Returns: list of all ids from similarity table."""
    with db.engine.connect() as connection:
        query = text("""
            SELECT id
              FROM similarity.similarity
          ORDER BY id
        """)
        result = connection.execute(query)
        return [r['id'] for r in result.fetchall()]


def add_index(index, num_ids, ids, batch_size):
    """Incrementally adds all items to an index, then builds
    and saves the index. *Note*: index must already be initialized.

    Args:
        indices: an initialized Annoy index.

        num_ids (int): total number of ids in the similarity.similarity
        table.

        ids (list): list of similarity.similarity.ids which should be
        added to the index.

        batch_size (int): the size of each batch of recording
        vectors being added in each increment.
    """
    num_added = 0

    # Fill empty rows in the database with placeholders
    # of the form [0, ..., 0]
    index = similarity.index_utils.add_empty_rows(index, ids)

    with db.engine.connect() as connection:
        batch_query = text("""
            SELECT id
                 , {}
              FROM similarity.similarity
             WHERE id IN :ids
          ORDER BY id
        """.format(index.metric_name))

        current_app.logger.info("Items added: {}/{} ({:.3f}%)".format(num_added, num_ids, float(num_added) / num_ids * 100))
        for sub_ids in chunks(ids, batch_size):
            # Get ids and vectors for specific metric in batches
            batch_result = connection.execute(batch_query, {"ids": tuple(sub_ids)})
            for row in batch_result:
                index.add_recording_with_vector(row["id"], row[index.metric_name])

            num_added += len(sub_ids)
            current_app.logger.info("Items added: {}/{} ({:.3f}%)".format(num_added, num_ids, float(num_added) / num_ids * 100))

        current_app.logger.info("Finished adding items. Building index...")
        index.build()
        current_app.logger.info("Saving index {}...".format(index.metric_name))
        index.save()


def get_metric_info(metric):
    """Gets the description and category for a given
    metric.

    Args:
        metric (str): The name of a similarity metric.

    Returns: metric info as a list in the form:
            [category, metric_name, metric_description]

    If no metric is available, a NoDataFoundException will
    be raised.
    """
    with db.engine.connect() as connection:
        query = text("""
            SELECT category
                 , metric
                 , description
              FROM similarity.similarity_metrics
             WHERE metric = :metric
        """)
        result = connection.execute(query, {"metric": metric})
        if not result.rowcount:
            raise db.exceptions.NoDataFoundException("There is no existing metric for the `metric` parameter.")
        return result.fetchone()


def add_metrics(batch_size=None):
    batch_size = batch_size or PROCESS_BATCH_SIZE
    lowlevel_count = count_all_lowlevel()

    with db.engine.connect() as connection:
        metrics = similarity.utils.init_metrics()

        # Collect and assign stats to metrics that require normalization
        for metric in metrics:
            db.similarity_stats.assign_stats(metric)

        sim_count = count_similarity()
        current_app.logger.info("Processed {} / {} ({:.3f}%)".format(sim_count,
                                                                     lowlevel_count,
                                                                     float(sim_count) / lowlevel_count * 100))

        max_id = get_max_similarity_id()
        while True:
            with connection.begin():
                result = get_batch_data(connection, max_id, batch_size)
                if not result:
                    break
                added_rows = 0
                all_rows = []
                for row in result:
                    data = bulk_transform_data_to_similarity(row, metrics=metrics)
                    all_rows.append(data)
                    added_rows += 1
                    max_id = max(max_id, row["id"])

                insert_similarity_bulk(connection, all_rows)

            sim_count += added_rows
            current_app.logger.info("Processed {} / {} ({:.3f}%)".format(sim_count,
                                                                         lowlevel_count,
                                                                         float(sim_count) / lowlevel_count * 100))


def get_batch_data(connection, max_id, batch_size):
    """Performs a query to collect highlevel models and lowlevel
    data for a batch of `batch_size` recordings.

    Args:
        connection: a connection to the database.
        max_id: get only items with an id greater than this
        batch_size: the number of recordings (rows) that should
        be collected in the query.

    Returns:
        If no rows are returned by the query, i.e. there are no
        available submissions, then None is returned.

        Otherwise, the result object of the query is returned.
    """
    batch_query = text("""
    SELECT llj.id
     , jsonb_build_object('mfcc', llj.data->'lowlevel'->'mfcc'->'mean',
                          'gfcc', llj.data->'lowlevel'->'gfcc'->'mean',
                          'bpm', llj.data->'rhythm'->'bpm',
                          'onset_rate', llj.data->'rhythm'->'onset_rate',
                          'key', jsonb_build_object('key_key', llj.data#>'{tonal,key_key}',
                                                    'key_scale', llj.data#>'{tonal,key_scale}')
                ) as ll_data
      , COALESCE(jsonb_object_agg(model.model, hlm.data)
         FILTER (
               WHERE model.model IS NOT NULL
               AND hlm.data IS NOT NULL), NULL)::jsonb AS hl_data
          FROM lowlevel_json llj
          LEFT JOIN highlevel_model AS hlm on hlm.highlevel = llj.id
          LEFT JOIN model
            ON model.id = hlm.model
         WHERE llj.id IN (
               SELECT id
               FROM lowlevel WHERE id > :max_id ORDER BY id LIMIT :batch_size)
      GROUP BY (llj.id)
      """)

    result = connection.execute(batch_query, {"max_id": max_id, "batch_size": batch_size})
    if not result.rowcount:
        return None
    return result


def insert_similarity(connection, id, vectors, metric_names):
    """Inserts a row of similarity vectors for a given lowlevel.id into
    the similarity table.

        Args:
            connection: a connection to the database.
            id: lowlevel.id to be submitted.
            vectors: list of metric vectors for a recording.
            metric_names: corresponding list of metric names.
    """
    params = {}
    params["id"] = id
    for name, vector in zip(metric_names, vectors):
        params[name] = list(vector)

    query = text("""
        INSERT INTO similarity.similarity (
                    id, %(names)s)
             VALUES (
                    :id, %(values)s)
        ON CONFLICT (id)
         DO NOTHING
    """ % {"names": ', '.join(metric_names),
           "values": ':' + ', :'.join(metric_names)})
    connection.execute(query, params)


def insert_similarity_bulk(connection, data):
    """Inserts a row of similarity vectors for a given lowlevel.id into
    the similarity table.

        Args:
            connection: a connection to the database.
            data (list): a list of items to add, each item a mapping of
            {metric names: vectors} for each metric to add
    """
    if not data:
        return

    metric_names = list(data[0].keys())
    query = text("""
        INSERT INTO similarity.similarity (
                    %(names)s)
             VALUES (
                    %(values)s)
    """ % {"names": ', '.join(metric_names),
           "values": ', '.join([':{}'.format(n) for n in metric_names])})
    connection.execute(query, data)


def count_similarity():
    # Get total number of submissions in similarity table
    with db.engine.connect() as connection:
        query = text("""
            SELECT COUNT(*)
              FROM similarity.similarity
        """)
        result = connection.execute(query)
        return result.fetchone()[0]


def get_max_similarity_id():
    # Get the highest id currently in the similarity table
    with db.engine.connect() as connection:
        query = text("""
            SELECT coalesce(max(id), 0)
              FROM similarity.similarity
        """)
        result = connection.execute(query)
        return result.fetchone()[0]


def submit_similarity_by_id(id):
    """Computes similarity metrics for a single recording specified
    by lowlevel.id, then inserts the metrics as a new row in the
    similarity table. Data will be collected before submission and
    metrics will be initialized.

    Args:
        id (int): lowlevel.id for desired submission.
    """
    metrics = similarity.utils.init_metrics()
    # Collect and assign stats to metrics that require normalization
    for metric in metrics:
        db.similarity_stats.assign_stats(metric)

    # Collect high and lowlevel data
    previous_id = id - 1
    with db.engine.connect() as connection:
        result = get_batch_data(connection, previous_id, 1)
        row = result.fetchone()
    if row:
        data = (row["ll_data"], row["hl_data"])
    else:
        raise db.exceptions.NoDataFoundException("No data found for the given id")

    vectors = []
    metric_names = []
    for metric in metrics:
        try:
            metric_data = metric.get_feature_data(data[0])
        except AttributeError:
            # High level metrics use models for transformation.
            metric_data = data[1]

        try:
            vector = metric.transform(metric_data)
        except ValueError:
            vector = [0] * metric.length()
        vectors.append(vector)
        metric_names.append(metric.name)

    with db.engine.connect() as connection:
        insert_similarity(connection, id, vectors, metric_names)


def bulk_transform_data_to_similarity(row, metrics):
    """Converts lowlevel and highlevel data for a batch of recordings
     and transforms it according to each metric, returning data to be
     inserted into the similarity table.

    Args:
        row: a row generated from get_batch_data, including id, ll_data, and hl_data

        metrics (list): a list of initialized metric classes, for which similarity
        vectors should be computed and submitted. Default is None, in which
        case base metrics will be initialized.
    """

    vectors = []
    metric_names = []
    results = {}
    for metric in metrics:
        try:
            metric_data = metric.get_feature_data(row['ll_data'])
        except AttributeError:
            # High level metrics use models for transformation.
            metric_data = row['hl_data']

        try:
            vector = metric.transform(metric_data)
        except ValueError:
            vector = [0] * metric.length()
        vectors.append(vector)
        metric_names.append(metric.name)
        results[metric.name] = vector

    results['id'] = row['id']
    return results


def submit_similarity_by_mbid(mbid, offset):
    """Computes similarity metrics for a single recording specified
    by (mbid, offset) combination, then inserts the metrics as a new
    row in the similarity table.

    Args:
        mbid (str): MBID for which similarity metrics should be computed.

        offset (int, non-negative): submission offset indicating the
        submission for the MBID whose metrics are being submitted.
    """
    id = db.data.get_ids_by_mbids([(mbid, offset)])[0]
    submit_similarity_by_id(id)


def get_metric_dimensionality(metric_name):
    """Get dimensionality of vectors for a metric in similarity table.

    Args:
        metric_name (str): name of metric for which dimensionality must
        be found.

    Returns:
        dimensionality (int): number of dimensions for metric vector.

    If no submissions are present in the similarity.similarity table,
    db.exceptions.NoDataFoundException will be raised.
    """
    if metric_name not in similarity.metrics.BASE_METRICS:
        raise db.exceptions.NoDataFoundException("No existing metric named \"{}\"".format(metric_name))
    with db.engine.connect() as connection:
        result = connection.execute("""
            SELECT *
              FROM similarity.similarity
             LIMIT 1
        """)
        try:
            dimensionality = len(result.fetchone()[metric_name])
            return dimensionality
        except TypeError:
            raise db.exceptions.NoDataFoundException("No existing similarity data.")


def get_similarity_by_mbid(mbid, offset):
    # Get a single row of the similarity table by (MBID, offset) combination
    with db.engine.connect() as connection:
        query = text("""
            SELECT *
              FROM similarity.similarity
             WHERE id = (
                SELECT id
                  FROM lowlevel
                 WHERE gid = :mbid
                   AND submission_offset = :offset )
        """)
        result = connection.execute(query, {"mbid": mbid, "offset": offset})
        if not result.rowcount:
            raise db.exceptions.NoDataFoundException(
                "No similarity metrics are computed for the given (MBID, offset) combination.")
        return result.fetchone()


def get_similarity_row_id(id):
    # Get a single row of the similarity table by lowlevel.id
    with db.engine.connect() as connection:
        query = text("""
            SELECT *
              FROM similarity.similarity
             WHERE id = :id
        """)
        result = connection.execute(query, {"id": id})
        if not result.rowcount:
            raise db.exceptions.NoDataFoundException("No similarity metrics are computed for the given id.")
        return result.fetchone()


def add_evaluation(user_id, eval_id, result_id, rating, suggestion):
    # Adds a row to the evaluation table.
    user_id = user_id or None
    with db.engine.begin() as connection:
        query = text("""
            INSERT INTO similarity.eval_feedback (user_id, eval_id, result_id, rating, suggestion)
                 VALUES (:user_id, :eval_id, :result_id, :rating, :suggestion)
            ON CONFLICT (user_id, eval_id, result_id)
             DO NOTHING
        """)
        connection.execute(query, {'user_id': user_id,
                                   'eval_id': eval_id,
                                   'result_id': result_id,
                                   'rating': rating,
                                   'suggestion': suggestion})


def submit_eval_results(query_id, result_ids, distances, params):
    """Submit a recording with its similar recordings and
    the parameters used to the similarity.eval_results table.

    Args:
        query_id: lowlevel.id which is the subject of a
                   query for similarity.
        result_ids: A list of lowlevel.ids), referencing the resultant
                   recordings.
        distances: A list of integers, the distance between each resultant
                   recording and the query recording.
        params: A list of form [<metric_name>, <n_trees>, <distance_type>]
    """
    with db.engine.connect() as connection:
        param_query = text("""
            SELECT id
              FROM similarity.eval_params
             WHERE metric = :metric
               AND n_trees = :n_trees
               AND distance_type = :distance_type
        """)
        result = connection.execute(param_query, {"metric": params[0],
                                                  "n_trees": params[1],
                                                  "distance_type": params[2]})
        if not result.rowcount:
            raise db.exceptions.NoDataFoundException('There are no existing eval params for the index specified.')
        params_id = result.fetchone()["id"]

        insert_query = text("""
            INSERT INTO similarity.eval_results (query_id, similar_ids, distances, params)
                 VALUES (:query_id, :similar_ids, :distances, :params)
            ON CONFLICT
          ON CONSTRAINT unique_eval_query_constraint
          DO UPDATE SET query_id = similarity.eval_results.query_id
              RETURNING id
        """)
        result = connection.execute(insert_query, {"query_id": query_id,
                                                   "similar_ids": result_ids,
                                                   "distances": distances,
                                                   "params": params_id})
        eval_result_id = result.fetchone()[0]
        return eval_result_id


def check_for_eval_submission(user_id, eval_id):
    """Checks similarity.eval_feedback for whether
    or not a submission already exists with the given
    user_id and eval_id.

    Args:
        user_id: user.id associated with user submitting
        the eval form.
        eval_id: similarity.eval_results.id, indicating the
        query and results associated with the eval form.

    Returns:
        If a submission exists, returns True. Otherwise,
        returns False.
    """
    with db.engine.connect() as connection:
        query = text("""
            SELECT *
              FROM similarity.eval_feedback
             WHERE user_id = :user_id AND eval_id = :eval_id
        """)
        result = connection.execute(query, {"user_id": user_id,
                                            "eval_id": eval_id})
        if not result.rowcount:
            return False
        return True
