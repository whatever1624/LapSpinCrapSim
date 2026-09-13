### Line Length

- Recommended limit of 140 characters so that 1 line fits on GitHub without a scroll bar on my 14-inch laptop ==CHECK THIS==
- Strict limit for docstrings and comments
- Relaxed limit for code and comments

---

### Global Constants

- Defined at the top of each Python file
- `UPPER_CASE`, does not need to follow the [Variable Conventions](Variable%20Conventions.md)

---

### Coordinates

- In lowercase as in `x`, `y`, `z` (even at the start of a sentence)

---

### Quotation Marks

- Single quotation marks for keys, substrings for conditionals etc.
- Double quotation marks for full strings like those in print statements

---

### Functions

- `snake_case` for all function names (follows the specific convention from [PEP 8](https://peps.python.org/pep-0008/#function-and-variable-names))
- Type hints for all functions
- Docstrings for all functions
- Newline for each argument (improves readability with type hints)

---

### Docstrings

- Follow [Line Length](#Line%20Length) as a strict limit
- Largely [Google formatted](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings) (see format example below)

```python
"""
1-sentence summary of the module/class/method/function

After a line break, give a more detailed description of the module/function if necessary
This can include example use cases
No trailing full stops, use newlines to separate sentences

Args:
    Arg1: Description of the argument, should contain the data type(s) expected if the type hints are unclear
        (particularly for lists and arrays)
        If this runs over a new line, use a tab indent for subsequent lines of the description
    Arg2: Again don't use full stops - separate sentences with newlines

Returns:
    Describe the return of the function, including the data types(s) expected
    If there are multiple returns, this should state "Returns a tuple of (a, b, ...)" and list the values returned
    similar to the Args section, such as below.

    return1: Description of the returned value, should contain the data type(s) expected

    return2: Similar things here.

    This section can also contain logic if the function return changes based on the arguments
    Note that blank lines are necessary for the documentation tooltip to render line breaks properly

Raises:
    Error1: Text printed out if the error is raised
    Error2: Same thing, if there are multiple possible errors that can be raised
"""
```

