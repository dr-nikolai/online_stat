# Single-Pass Online Statistics Algorithms

## Suppliment to the article:

[Single-Pass Online Statistics Algorithms](http://www.numericalexpert.com/articles/single_pass_stat/ ), 
by Nikolai Shokhirev, 2013

### Author

[Nikolai Shokhirev](http://www.numericalexpert.com/contact.php) 

### Dependencies

numpy

### Usage example

```python
# Test data
X = np.array([0.2,1.0,1.4,2.0,2.7,3.8,5.4,7.5,11,15],dtype=float)
```

```python
    # Create processing object
    n = 3 # window size
    ws =  = WindowedStat(n)    
    print('Mean,  M2,  M3,  M4')
    for x in X:
        ws.push(x)
        print(ws.m, ws.m2, ws.m3, ws.m4)
```

Output:

    Mean,       M2,  M3,  M4
    0.2           0.0 0.0 0.0
    0.6           0.16            1.38777878e-17  0.0256
    0.86666666667 0.248888888889 -0.0474074074074 0.0929185185185
    1.46666666667 0.168888888889  0.0165925925926 0.0427851851852
    2.03333333333 0.282222222222  0.0140740740741 0.119474074074
    2.83333333333 0.548888888889  0.107407407407  0.451918518519
    3.96666666667 1.22888888889   0.302592592593  2.26525185185
    5.56666666667 2.29555555556   0.569259259259  7.90436296296
    7.96666666667 5.33555555556   3.63325925926 42.7022296296
    11.1666666667 9.38888888889   2.34259259259 132.226851852


## License

#### (The MIT License)

Copyright (c) 2014 Bill Gooch

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
        distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

        The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
        EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
        IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
        TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.









