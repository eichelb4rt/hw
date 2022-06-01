class adouble(float):
    def __init__(self, original, derivative=1.0):
        self.original = original,
        self.derivative = derivative
    
    def __add__(self, other: 'adouble') -> 'adouble':
        return adouble(self.original + other.original, self.derivative + other.derivative)
    
    def __mul__(self, other: 'adouble') -> 'adouble':
        return adouble(self.original * other.original, self.original * other.derivative + self.derivative * other.original)
    
    def __truediv__(self, other: 'adouble') -> 'adouble':
        return adouble(self.original / other.original, (self.original * other.derivative - self.derivative * other.derivative) / other.original**2)
    
    def __add__(self, other: 'adouble') -> 'adouble':
        return adouble(self.original - other.original, self.derivative - other.derivative)


    