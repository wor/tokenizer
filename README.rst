Tokenizer
=========

Python regular expression based tokenizer library. More info and use cases to
come..

Contains also basic recursive parser (rparser.py_) module.

An example of simple usage can be found from: `Desktop entry file tokenizer`_

If this Python "re" library error is experienced, then optional dependency
`regex`_ should be used::

    File "/usr/lib/python3.3/sre_compile.py", line 505, in compile
        "sorry, but this version only supports 100 named groups"
        AssertionError: sorry, but this version only supports 100 named groups


Dependencies
------------

Optional: `regex`_


.. _`Desktop entry file tokenizer`: https://github.com/wor/desktop_file_parser/blob/master/src/wor/desktop_file_parser/tokenizer.py
.. _`rparser.py`: https://github.com/wor/tokenizer/blob/master/src/wor/rparser.py
.. _`regex`: https://pypi.python.org/pypi/regex
