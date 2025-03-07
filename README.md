# np<ins>s</ins>typing

#### NumPy <ins>s</ins>trict <ins>s</ins>hape typing – Module for special Numpy use cases.

## Overview, history and current situation regarding this

### What NumPy does

Although the NumPy developers have basically specified the data type to be stored in the array with the dtype attribute, in the most general case any arbitrary Python object (dtype ‘O’), but never a parameter and associated methodology for restricting the shape. Ok: NumPy is intended for mathematical calculations. For this purpose, a special restriction of the shape of arrays does not appear to be necessary. In the context of application and module programming for special libraries or entire applications, it can be useful for quality assurance purposes to restrict NumPy arrays as far as possible in terms of data typing **and** shape sizes.

### Typing principles in Python

There are two basic ways to implement a type in Python:

* static typing (works offline with special development tools (e.g. mypy and Pyright))
* type classes with check methods like isinstance and asserts and other individial solutions for data content and type checks of data during the runtime

All type classification models in Python are unable to specify the limits of composite fields. You can specify the dimension of the fields. You can specify which data type is stored in these fields. But not whether limits are set in the individual dimensions of the single field dimension **size**. If we now want to typify this field size restriction, as nptyping has demonstrated for NumPy version 1.x (see next section), we have to decide whether we want to enable static type checking for it or implement the whole thing dynamically, i.e. at runtime.

### nptyping (withou the 's' between 'np' and 'typing')

When using static typing, conformity to the officially supported typing clauses and the implementations in mypy or pyright is necessary. The [nptyping Modul](https://pypi.org/project/nptyping/), developed by [Ramon Hagenaars](https://stackoverflow.com/users/2169290/r-h) for Python starting with version 3.7 upwards and Numpy 1.x, has gone this way. [This](https://stackoverflow.com/a/72585748) is probably the most well-known thread of recent years, in which Ramon presented the various options and his tool nptyping in a (so far last; as of 03/2025) reply. With the version jump of Numpy to version 2, Ramon unfortunately did not continue the development (for the time being?). [Jake](https://stackoverflow.com/users/534674/jake-stevens-haas) wrote a [comment in the same thread](https://stackoverflow.com/a/77907698) that shape typing was added to Numpy. He referenced [this](https://github.com/numpy/numpy/pull/26081). But I could not recognize that mypy and pyright can observ the shape of NumPy arrays until now (03/2025).

### This module in comparison

I was thinking about how an alternative module could handle the shape of NumPy arrays. I want a ‘small’ solution, which means that I don't want to initiate a parallel development at mypy and pyright to avoid lengthy planning discussions and complex maintenance scenarios. It quickly became clear that static type checking, as with nptyping, is insufficient. For more flexible use, the type checkers would have to be adapted accordingly and provided with intelligence. Since I do not want this dependency, the only solution left for the time being is a dynamic shape check, i.e. at runtime.

To find a compromise, the implementation of type checking should be based on assert statements, or rather the \_\_debug\_\_ parameter. So a time critical application can use the '-O' (optimisation) parameter to optimize the code for speed and switch off type checking in this case.

### An additional focus

What about NumPy arrays with non-standard Numpy data types? For such data, Numpy uses the ‘Object’ type, abbreviated in Numpy as ‘O’. This means that any Python data type can be stored in NumPy arrays, provided that it makes sense in terms of the NumPy methods. Perhaps the most common data types in this way are probably datetime.datetime and datetime.timedelta. Although this makes little sense for reasons of efficiency, since the 4-byte NumPy data types datetime64 and timedelta64 are available for this purpose and are also recognised by matplotlib.pyplot and displayed correctly, it is very convenient in many applications to stick with the basic Python data types for times and time intervals.

For the use of special non-NumPy data types, **this module is also intended to combine the type checking of such data types together with the standard dtype type checking (issubdtype) into a single type check**.

If you are already using **datetime64 and timedelta64, you may have encountered the problem of how to convert this data type error-free to and from datetime.datetime and datetime.timeldata.** The problem is that so far (NumPy 2.2), the datetime64 and timedelta64 data types are internally parameterised with a base time resolution. Within NumPy, this is taken into account without any problems. However, if you extract the value and try to convert it into a real point in time or a real period of time, you get integer values that depend on this time base! The problem can be solved by a clever formulation of the code line. Since the developer of npstyping has repeatedly experienced such conversions as a useless waste of time in clarifying the conversion problem, this should also be taken into account in npstyping.

# np<ins>s</ins>typing – strictly shaped typing for NumPy arrays

In summary, this module has the following aims:

* Primary: Observe / check the shape of arrays during processing.
* Secondary 1: Extend dtype checks to ndarray-dtype 'object' cases (dtype = 'O')
* Secondary 2: Simplify the conversion from numpy.datetime64 and numpy.timedelta64 arrays.

# State

## Work in progress

This means that the main goal is in an first usable state for intensive testing and in the ‘Main’ branch. It will be published during further development. The secondary goals have been solved in principle, but have not yet been added to this repository.
