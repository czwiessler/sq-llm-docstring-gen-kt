from flask.cli import FlaskGroup
import click

import webserver
import similarity.index_utils
from similarity.index_model import AnnoyModel
import db
import db.similarity
import db.similarity_stats

NORMALIZATION_SAMPLE_SIZE = 10000
ADD_METRICS_BATCH_SIZE = 10000
ADD_INDEX_BATCH_SIZE = 100000

cli = FlaskGroup(add_default_commands=False, create_app=webserver.create_app)


@cli.command(name="init")
@click.option("--distance-type", "-d", default='angular', help="Method of calculating distance for index.")
@click.option("--n-trees", "-n", type=int, default=10, help="Number of trees for building index. \
                                                            Tradeoff: more trees gives more precision, \
                                                            but takes longer to build.")
@click.option("--force", "-f", default=False, help="Remove existing stats before computing.")
@click.option("--sample-size", "-s", type=int, default=NORMALIZATION_SAMPLE_SIZE,
              help="Override normalization lowlevel data sample size. Must be >= 1% of lowlevel_json entries.")
@click.option("--batch-size", "-b", type=int, default=ADD_METRICS_BATCH_SIZE, help="Override processing batch size.")
@click.pass_context
def init(ctx, batch_size, sample_size, force, n_trees, distance_type):
    """Initialization command for the similarity engine.
    The following steps will occur:
        1. Compute global stats required for similarity
           using a sample of the lowlevel_json table.
        2. Compute base metrics for all recordings in
           the lowlevel table, inserting these values
           in the similarity.similarity table.
        3. Initialize, build, and save indices for each
           of the metrics including every recording in
           the similarity.similarity table
    """
    ctx.invoke(compute_stats, sample_size=sample_size, force=force)
    ctx.invoke(add_metrics, batch_size=batch_size)
    ctx.invoke(add_indices, batch_size=batch_size, n_trees=n_trees, distance_type=distance_type)


@cli.command(name="compute-stats")
@click.option("--force", "-f", default=False, help="Remove existing stats before computing.")
@click.option("--sample-size", "-s", type=int, default=NORMALIZATION_SAMPLE_SIZE,
              help="Override normalization lowlevel data sample size. Must be >= 1% of lowlevel_json entries.")
def compute_stats(sample_size, force):
    """Computes the mean and standard deviation for
    lowlevel features that are associated with the
    normalized metrics.

    Stats are computed using a sample of items from the
    lowlevel_json table, configured using `--sample-size`.

    Adds these statistics to the similarity.similarity_stats
    table with the corresponding metric.

    A list of normalized metrics:
        - MFCCs
        - Weighted MFCCs
        - GFCCs
        - Weighted GFCCs
    """
    click.echo("Computing stats...")
    db.similarity_stats.compute_stats(sample_size, force=force)
    click.echo("Finished!")


@cli.command(name="add-metrics")
@click.option("--batch-size", "-b", type=int, default=ADD_METRICS_BATCH_SIZE, help="Override processing batch size.")
def add_metrics(batch_size):
    """Computes all 12 base metrics for each recording
    in the lowlevel table, inserting these values in
    the similarity.similarity table.

    Requires fetching lowlevel data and the highlevel
    models for each recording.

    Args:
        batch_size: integer, number of recordings that
        should be added on each iteration.
        Suggested value between 10 000 and 20 000.
    """
    click.echo("Adding all metrics...")
    db.similarity.add_metrics(batch_size)
    click.echo("Finished adding all metrics, exiting...")


@cli.command(name='add-index')
@click.argument("metric")
@click.option("--distance_type", "-d", default='angular', help="Method of measuring distance between metric vectors")
@click.option("--n_trees", "-n", type=int, default=10, help="Number of trees for building index. \
                                                            Tradeoff: more trees gives more precision, \
                                                            but takes longer to build.")
@click.option("--batch_size", "-b", type=int, default=ADD_INDEX_BATCH_SIZE, help="Size of batches")
def add_index(metric, batch_size, n_trees, distance_type):
    """Creates an annoy index for the specified metric using the given params.
    This operates by creating a special case of `db.similarity.add_indices`,
    where the list of indices only contains one index.

    *NOTE*: Using this command overwrites any existing index with the
    same parameters.
    """
    click.echo("Gathering ids to insert...")
    num_ids = db.similarity.get_similarity_count()
    ids = db.similarity.get_similarity_ids()
    click.echo("Initializing index...")
    index = AnnoyModel(metric, n_trees=n_trees, distance_type=distance_type)
    click.echo("Adding index: {}".format(metric))
    db.similarity.add_index(index, num_ids, ids, batch_size=batch_size)
    click.echo("Done!")


@cli.command(name='add-indices')
@click.option("--distance-type", "-d", default='angular')
@click.option("--n-trees", "-n", type=int, default=10, help="Number of trees for building index. \
                                                            Tradeoff: more trees gives more precision, \
                                                            but takes longer to build.")
@click.option("--batch_size", "-b", type=int, default=ADD_INDEX_BATCH_SIZE, help="Size of batches")
def add_indices(batch_size, n_trees, distance_type):
    """Creates an annoy index then adds all recordings to the index,
    for each of the base metrics.

    *NOTE*: Using this command overwrites any existing index with the
    same parameters.
    """
    click.echo("Gathering ids to insert...")
    num_ids = db.similarity.get_similarity_count()
    ids = db.similarity.get_similarity_ids()
    click.echo("Initializing indices...")
    indices = similarity.index_utils.initialize_indices(n_trees=n_trees, distance_type=distance_type)
    for index in indices:
        click.echo("Adding index: {}".format(index.metric_name))
        db.similarity.add_index(index, num_ids, ids, batch_size=batch_size)
    click.echo("Finished adding all indices. Exiting...")


@cli.command(name='remove-index')
@click.argument("metric")
@click.option("--distance_type", "-d", default='angular', help="Method of measuring distance between metric vectors.")
@click.option("--n_trees", "-n", type=int, default=10, help="Number of trees for building index. \
                                                            Tradeoff: more trees gives more precision, \
                                                            but takes longer to build.")
def remove_index(metric, n_trees, distance_type):
    """Removes the index with the specified parameters, if it exists.

        Note that each index is built with a distinct number of trees,
        metric, and distance type.
    """
    click.echo("Removing index: {}".format(metric))
    similarity.index_utils.remove_index(metric, n_trees=n_trees, distance_type=distance_type)
    click.echo("Finished.")


@cli.command(name='remove-indices')
@click.option("--distance_type", "-d", default='angular', help="Method of measuring distance between metric vectors.")
@click.option("--n_trees", "-n", type=int, default=10, help="Number of trees for building index. \
                                                            Tradeoff: more trees gives more precision, \
                                                            but takes longer to build.")
def remove_indices(n_trees, distance_type):
    """Removes indices for each of the following metrics, if they
    exist with the specified parameters."""
    metrics = ["mfccs",
               "mfccsw",
               "gfccs",
               "gfccsw",
               "key",
               "bpm",
               "onsetrate",
               "moods",
               "instruments",
               "dortmund",
               "rosamerica",
               "tzanetakis"]

    for metric in metrics:
        click.echo("Removing index: {}".format(metric))
        similarity.index_utils.remove_index(metric, n_trees=n_trees, distance_type=distance_type)
    click.echo("Finished.")
