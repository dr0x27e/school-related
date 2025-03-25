-- Create two bascases 0 and 1
fib :: Int -> Int
fib 0 = 0
fib 1 = 1
fib n = fib (n - 1) + fib (n - 2)

main = do
    putStrLn (show (fib 6))
