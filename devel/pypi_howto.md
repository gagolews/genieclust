Versions:

1.2.0.dev1  # Development release
1.2.0a1     # Alpha Release
1.2.0b1     # Beta Release
1.2.0rc1    # Release Candidate
1.2.0       # Final Release


# platform wheels:
~~python3 setup.py bdist_wheel~~


Github actions - cibuildwheel - run manually

# source distribution:

rm -f dist/*
python3 setup.py sdist


# upload to pypi:

twine upload dist/*
