# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022-2024)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The test injects a resizer-script into an iframe. When the app is
interacted with and the markdown elements are added, the iframe is resized automatically
to wrap the content.
"""

import streamlit as st

x = st.slider("Enter a number", 0, 20, 0)

for _ in range(x):
    st.write("Hello example")
