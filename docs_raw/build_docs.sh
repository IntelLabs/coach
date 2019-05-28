#!/bin/bash

echo "installing requirements..."

pip3 install Sphinx
pip3 install recommonmark
pip3 install sphinx_rtd_theme
pip3 install sphinx-autobuild
pip3 install sphinx-argparse

echo "Making docs..."

make html

echo "Copying new docs into coach/docs/"

cp source/_static/css/custom.css build/html/_static/css/
rm -rf ../docs/
mkdir ../docs
touch ../docs/.nojekyll
cp -R build/html/* ../docs/
rm -r build

echo "Finished!"