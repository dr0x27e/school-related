fib :: Int -> Int
fib x = fibs !! x
    where
        fibs :: [Int]
	fibs = 0 : 1 : zipWith (+) fibs (tail fibs)


main = do
    putStrLn (show (fib 5))
