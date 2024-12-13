# -*- coding: utf-8 -*-
import json
import os

from six import PY2
from nose.tools import assert_in
import requests
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from hyperdash import monitor
from hyperdash.constants import API_KEY_NAME
from hyperdash.constants import API_NAME_JUPYTER
from hyperdash.constants import get_hyperdash_logs_home_path_for_job
from hyperdash.constants import get_hyperdash_version
from hyperdash.constants import VERSION_KEY_NAME
from mocks import init_mock_server


server_sdk_headers = []


class TestJupyter(object):
    def setup(self):
        global server_sdk_headers
        server_sdk_headers = []

    @classmethod
    def setup_class(_cls):
        request_handle_dict = init_mock_server()

        def sdk_message(response):
            # Store headers so we can assert on them later
            if PY2:
                server_sdk_headers.append(response.headers.dict)
            else:
                server_sdk_headers.append(response.headers)

            # Add response status code.
            response.send_response(requests.codes.ok)

            # Add response headers.
            response.send_header('Content-Type', 'application/json; charset=utf-8')
            response.end_headers()

            # Add response content. In this case, we use the exact same response for
            # all messages from the SDK because the current implementation ignores the
            # body of the response unless there is an error.
            response_content = json.dumps({})

            response.wfile.write(response_content.encode('utf-8'))

        request_handle_dict[("POST", "/api/v1/sdk/http")] = sdk_message

    def test_monitor_cell(self):
        expected_logs = [
            "Beginning machine learning...",
            "Still training...",
            "Done!",
            # Handle unicode
            "字"
        ]

        with open("./tests/jupyter_test_file.py.ipynb") as f:
            nb = nbformat.read(f, as_version=4.0)

        ep = ExecutePreprocessor(timeout=5000, kernel_name="python")
        result = ep.preprocess(nb, {})

        # Accumulate all output from cell 2
        all_stdout = ""
        second_cell_output = result[0]["cells"][1]["outputs"]
        for output in second_cell_output:
            if output.name == "stdout":
                all_stdout = all_stdout + output.text

        # Assert on all output from cell2
        for log in expected_logs:
            if PY2:
                assert_in(log, all_stdout.encode("utf-8"))
                continue
            assert_in(log, all_stdout)

        # Verify that variables declared in previous cells can be affected
        third_cell_output = result[0]["cells"][2]["outputs"]
        assert third_cell_output[0].text == "a=1\n"

        # Make sure correct API name / version headers are sent
        assert server_sdk_headers[0][API_KEY_NAME] == API_NAME_JUPYTER
        assert server_sdk_headers[0][VERSION_KEY_NAME] == get_hyperdash_version()

        # Make sure logs were persisted
        log_dir = get_hyperdash_logs_home_path_for_job("test_jupyter")
        latest_log_file = max([
            os.path.join(log_dir, filename) for
            filename in
            os.listdir(log_dir)
        ], key=os.path.getmtime)
        with open(latest_log_file, 'r') as log_file:
            data = log_file.read()
            for log in expected_logs:
                assert_in(log, data)
        os.remove(latest_log_file)
