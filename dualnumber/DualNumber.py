from numbers import Number
import math


class DualNumber:

    def __init__(self, real, dual):
        self.a = real
        self.b = dual

    def _add(self, num2):
        if isinstance(num2, DualNumber):
            return DualNumber(self.a + num2.a, self.b + num2.b)
        elif isinstance(num2, Number):
            return DualNumber(self.a + num2, self.b)
        else:
            raise TypeError("unsupported operand type(s) for +: '{num2.type}'" \
                " and 'DualNumber'")
    
    def __add__(self, num2):
        return self._add(num2)
    
    def __radd__(self, num2):
        return self._add(num2)

    def _sub(self, num2, self_first=True):
        if self_first and isinstance(num2, DualNumber):
            return DualNumber(self.a - num2.a, self.b - num2.b)
        elif self_first and isinstance(num2, Number):
            return DualNumber(self.a - num2, self.b)
        elif not self_first and isinstance(num2, Number):
            return DualNumber(num2 - self.a , -1 * self.b)
        else:
            raise TypeError("unsupported operand type(s) for -: '{num2.type}'" \
                " and 'DualNumber'")
    
    def __sub__(self, num2):
        return self._sub(num2)
    
    def __rsub__(self, num2):
        return self._sub(num2, False)
    
    def _mul(self, num2):
        if isinstance(num2, DualNumber):
            return DualNumber(self.a * num2.a, self.a * num2.b + self.b * num2.a)
        elif isinstance(num2, Number):
            return DualNumber(self.a * num2, self.b * num2)
        else:
            raise TypeError("unsupported operand type(s) for *: '{num2.type}'" \
                " and 'DualNumber'")
    
    def __mul__(self, num2):
        return self._mul(num2)
    
    def __rmul__(self, num2):
        return self._mul(num2)

    def _div(self, num2, self_Nx=True):
        # TODO - (a + b ep) / (0 + d ep)
        if self_Nx:
            if isinstance(num2, Number):
                if num2 == 0:
                    raise ZeroDivisionError("Attempt to divide by zero")
            if isinstance(num2, DualNumber):
                if num2.a == 0:
                    raise ZeroDivisionError("Attempt to divide by zero")

        if self_Nx and isinstance(num2, DualNumber):
            return DualNumber(self.a / num2.a, (self.b * num2.a - self.a * num2.b) / (num2.a * num2.a))
        elif self_Nx and isinstance(num2, Number):
            return DualNumber(self.a / num2, self.b / num2)
        elif not self_Nx and isinstance(num2, Number):
            return DualNumber(num2 / self.a , -1 * self.b * num2 / (self.a * self.a))
        else:
            raise TypeError("unsupported operand type(s) for /: '{num2.type}'" \
                " and 'DualNumber'")
    
    def __truediv__(self, num2):
        return self._div(num2)

    def __rtruediv__(self, num2):
        return self._div(num2, False)

    def __div__(self, num2):
        return self._div(num2)
    
    def __rdiv__(self, num2):
        return self._div(num2, False)

    def _pow(self, num2, self_base=True):
        if self_base and isinstance(num2, DualNumber):
            a = self.a ** num2.a
            b = (a / self.a) * (self.a * num2.b * math.log(self.a)  + num2.a * self.b)
            return DualNumber(a, b)
        elif self_base and isinstance(num2, Number):
            a = self.a ** num2
            b = (a / self.a) * self.b * num2
            return DualNumber(a, b)
        elif not self_base and isinstance(num2, Number):
            a = num2 ** self.a
            b = a * self.b * math.log(num2)
            return DualNumber(a ,b )
        else:
            raise TypeError("Unsupported operand type(s) for pow: '{num2.type}'" \
                " and 'DualNumber'")
    
    def __pow__(self, num2):
        return self._pow(num2)

    def __rpow__(self, num2):
        return self._pow(num2, False)

    def __cmp__(self, num2):
        if isinstance(num2, DualNumber):
            return self.a - num2.a
        elif isinstance(num2, Number):
            return self.a - num2
        else:
            raise TypeError("unsupported operand type(s) for comparision: '{num2.type}'" \
                " and 'DualNumber'")
    
    def __str__(self):
        s = self.b > 0
        s = '+' if s else '-'
        b = self.b if abs(self.b) != 1 else ''
        return "{} {} {}ɛ".format(self.a, s, b)