# Coach Documentation

Coach uses Sphinx with a Read The Docs theme for its documentation website.
The website is hosted on GitHub Pages, and is automatically pulled from the repository through the built docs directory.

To build the documentation website locally, first install the following requirements:

```
pip install Sphinx
pip install recommonmark
pip install sphinx_rtd_theme
pip install sphinx-autobuild
pip install sphinx-argparse
```

Then there are two option to build:
1. Build using the make file (recommended). Run from within the `docs_raw` directory:

```
make html
cp source/_static/css/custom.css build/html/_static/css/
rm -rf ../docs/
mkdir ../docs
touch ../docs/.nojekyll
cp -R build/html/* ../docs/
```

2. Build automatically after every change while editing the files:

```
sphinx-autobuild source build/html
```
