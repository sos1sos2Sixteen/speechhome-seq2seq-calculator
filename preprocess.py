import random 

class ArithmeticIter(): 
    def __init__(self, max_digits: int = 3) -> None: 
        self.max_digits = max_digits
    
    def get_num(self, ) -> int: 
        n = random.randint(1, self.max_digits)
        return int(
            ''.join((str(random.randint(0, 9)) for _ in range(n)))
        )
    
    def __iter__(self, ): return self

    def __next__(self, ): 
        a = self.get_num()
        b = self.get_num() 
        if random.random() > 0.5: 
            c = a + b
            return f'{a}+{b}', f'{c}'
        else: 
            c = a - b 
            return f'{a}-{b}', f'{c}'
        
if __name__ == '__main__': 
    for i, (q, a) in zip(range(10000), ArithmeticIter(3)): 
        print(f'{q}|{a}')
    