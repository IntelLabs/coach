#
# Copyright (c) 2017 Intel Corporation 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/usr/bin/env python3

from distutils.core import setup

long_description = \
"""This extension adds math formulas support to Python-Markdown_
(works with version 2.6 or newer).

.. _Python-Markdown: https://github.com/waylan/Python-Markdown

You can find the source on GitHub_.
Please refer to the `README file`_ for details on how to use it.

.. _GitHub: https://github.com/mitya57/python-markdown-math
.. _`README file`: https://github.com/mitya57/python-markdown-math/blob/master/README.md
"""

setup(name='python-markdown-math',
      description='Math extension for Python-Markdown',
      long_description=long_description,
      author='Dmitry Shachnev',
      author_email='mitya57@gmail.com',
      version='0.2',
      url='https://github.com/mitya57/python-markdown-math',
      py_modules=['mdx_math'],
      license='BSD')
