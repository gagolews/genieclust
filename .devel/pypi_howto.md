Versions:

1.0.9.9001  # development release
1.1.0       # major release


# platform wheels:

```
~~python3 setup.py bdist_wheel~~
```

Github actions - cibuildwheel - run manually

# source distribution:

```
rm -f dist/*
python3 setup.py sdist
```

(copy files)


# upload to pypi:

```
twine upload dist/*
```

