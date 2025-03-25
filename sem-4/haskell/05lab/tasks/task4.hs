{-
 - fibs = 0 : 1 : next fibs
 -    where
 -        next (a : t@(b:_)) : next t
 - 
 - This defines an infinite list of lazy evaluations
 - haskell is inherently lazy and only computes on demand therefor we can define
 - with constructor operator ":" an infenite list like this 0 : 1 : next fibs
 - This works the same way as [1..n]
 -
 - A is the first element in the list, t@ means tail and as-patterned which basically
 - sets t as a variable aswell.
 -
 - Example fibs 4:
 -
 - 1: a = 0, b = 1, t = next fibs (on the new array)
 - 2: a = 1, b = 1, t = next fibs (on the new array)
 - 3: a = 1, b = 2, t = next fibs (on the new array)
 - 4: a = 2, b = 3, t = next fibs (on the new array)
 -
 - Alternativly and more simply we dont need to use t@ we can just utilize the _ in the patternmatching
 - simply assign it instead of throwing it away: next (a : b : bs) : next bs
 - Or so i belive and also chatgpt when asked, but in pratice it does not seem to work.
-}

fib :: Int -> Int
fib n = fibs !! n
  where
    fibs = 0 : 1 : next fibs
      where
        next (a : t@(b:_)) = (a + b) : next t


fib2 :: Int -> Int
fib2 n = fibs !! n
  where
    fibs = 0 : 1 : next fibs
      where
        next (a : b : bs) = (a + b) : next bs
	next _ = []


main :: IO ()
main = do
    putStrLn (show (fib 5))
