#!/usr/bin/env python3
# Copyright 2017 Karl Sundequist Blomdahl <karl.sundequist.blomdahl@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=C0103, C0301

"""
xxx
"""

import argparse
import base64
from datetime import datetime, timezone
import http.client
import json
import gzip
import re
import struct
import sys
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs


DT = re.compile(r'DT\[([^\]]*)\]')

def time_from_sgf(sgf):
    """ Retrieve the creation time of the given SGF """
    m = DT.search(sgf)

    if m:
        d = None

        try:
            # check for our extended date-time format first (which
            # includes the time with a higher precision)
            if not d:
                d = datetime.strptime(m.group(1), '%Y-%m-%dT%H:%M:%S%z')
        except ValueError:
            pass

        try:
            # this is the Tygem SGF format
            if not d:
                d = datetime.strptime(m.group(1), '%Y.%m.%d %H:%M')
        except ValueError:
            pass

        try:
            # this is the standard SGF format
            if not d:
                d = datetime.strptime(m.group(1), '%Y-%m-%d')
        except ValueError:
            pass

        if d:
            return d.astimezone(timezone.utc) if d.tzinfo else d

    return datetime.utcnow()

def read_tfrecord(input):
    """ Read a binary TFRecord from the given input stream, and return the raw
    bytes. """

    # Each TFRecord has the following format, we will only read the first 8
    # bytes in order to fetch the `length`, and then assume the rest of the
    # record looks correct.
    #
    # ```
    # uint64 length
    # uint32 masked_crc32_of_length
    # byte   data[length]
    # uint32 masked_crc32_of_data
    # ```
    #
    length_bytes = b''

    while len(length_bytes) < 8:
        buf = input.read(8 - len(length_bytes))
        if not buf:
            return None

        length_bytes += buf

    length = struct.unpack('<Q', length_bytes)[0] + 8
    remaining = b''

    while len(remaining) < length:
        remaining += input.read(length - len(remaining))

    return length_bytes + remaining

parser = argparse.ArgumentParser(description='Upload data from standard input')
parser.add_argument('--sgf', action='store_true')
parser.add_argument('--tfrecord', action='store_true')
parser.add_argument('--bytes', type=int, nargs='?')
parser.add_argument('--date', type=int, nargs='?', help='Override the default date')
parser.add_argument('host', nargs=1)

args = parser.parse_args(sys.argv[1:])
if not args.sgf and not args.bytes and not args.tfrecord:
    parser.print_help()
    quit()

# establish a connection to the provided host
url = urlparse(args.host[0])
query = parse_qs(url.query)
rest = http.client.HTTPConnection(url.netloc, timeout=60)

while not sys.stdin.closed:
    if args.sgf:
        payload = sys.stdin.readline().strip()
        timestamp = time_from_sgf(payload)

        # convert the payload to bytes
        payload = payload.encode('utf-8')
    elif args.tfrecord:
        payload = read_tfrecord(sys.stdin.buffer)
        timestamp = datetime.utcnow()
    elif args.bytes > 0:
        payload = sys.stdin.buffer.read(args.bytes)
        timestamp = datetime.utcnow()
    else:
        payload = sys.stdin.buffer.read()
        timestamp = datetime.utcnow()

    # quit if EOF
    if not payload:
        break

    # upload to the api
    body = {}

    for key, value in query.items():
        body[key] = value[0]

    body['created_at'] = args.date or timestamp.isoformat()
    body['data'] = base64.b64encode(payload).decode('ascii')

    #
    def try_to_request(count=0):
        try:
            rest.request(
                'POST',
                urlunparse((
                    '',  # schema
                    '',  # netloc
                    url.path,  # path
                    url.params,  # params
                    '',  # query
                    url.fragment  # fragment
                )),
                json.dumps(body),
                { 'Content-Type': 'application/json'}
            )

            # check that we succeeded
            resp = rest.getresponse()
            resp.read()

            return resp
        except Exception as e:
            if count < 3:
                return try_to_request(count + 1)
            else:
                print(e)

                return None

    resp = try_to_request()

    if not resp or resp.status != 200:
        print(payload)
