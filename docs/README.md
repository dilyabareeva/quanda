Run the following commands to generate the API documentation:
```
cd docs
mkdir -p source/_static
make clean
make rst
make html
```

Note: always run `make clean` before `make html` to avoid stale cache issues with the sidebar navigation.

To preview the docs locally, serve the built HTML:
```
python -m http.server -d build/html 8000
```
Then open `http://localhost:8000` in your browser.
