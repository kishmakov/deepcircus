# cpp/validation

The standalone `validation` executable. It reconstructs gate schemes from
boolean functions and checks the result using the project's exact solvers.

The implementation contains the scheme representation, scoring, search, and
executable driver. It depends only on `common`.
