-- Recursivly:
mfact1 :: Int -> Int
mfact1 0 = 1
mfact1 x = x * mfact1 (x - 1)

-- Non recursive:
mfact2 :: Int -> Int
mfact2 x = foldr (*) x [2..(x-1)]

main = do
    putStrLn ( show (mfact1 5))
    putStrLn ( show (mfact2 5))


