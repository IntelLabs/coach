installation
=============
1. install mkdocs by following the instructions here - 
	http://www.mkdocs.org/#installation
2. install the math extension for mkdocs 
	sudo -E pip install python-markdown-math
3. install the material theme
	sudo -E pip install mkdocs-material

to build the documentation website run:
- mkdocs build
- python fix_index.py

this will create a folder named site which contains the documentation website
