import inspect

import ramanchada2 as rc2


# Get all methods or functions with decorators
def get_decorated_functions(cls):
    decorated_functions = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):

        if hasattr(method, "__wrapped__"):

            doc = method.__doc__  # Get the docstring (None if not available)
            decorated_functions.append((name, "" if doc is None else doc))
    return decorated_functions


def generate_decorated_function_docs_markdown(cls, filename="spectrum_functions.md"):
    """
    Generates a markdown file that lists decorated functions and their docstrings.
    """
    with open(filename, "w") as f:
        f.write(f"# Decorated Functions in {cls.__name__}\n\n")

        decorated_funcs_with_docs = get_decorated_functions(cls)

        for func_name, doc in decorated_funcs_with_docs:
            f.write(f"### Function: `{func_name}`\n\n")
            f.write(f"**Docstring:** {doc if doc else 'No docstring available'}\n\n")
            f.write("---\n\n")


generate_decorated_function_docs_markdown(rc2.spectrum.Spectrum)
