
count1 :: Int -> [Int] -> Int
count1 x y = sum (map check y)
    where
        check :: Int -> Int
	check n = if n == x then 1 else 0

count2 :: Int -> [Int] -> Int
count2 x y = sum $ map (\n -> if n == x then 1 else 0) y


count3 :: Int -> [Int] -> Int
count3 x = foldr (\n acc -> if n == x then acc + 1 else acc) 0

count4 :: Int -> [Int] -> Int
count4 x y = sum (zipWith (\a b -> if a == b then 1 else 0) y (repeat x))

main = do
    putStrLn (show (count1 2 [1, 2, 2, 3, 4, 2, 5]))
    putStrLn (show (count2 2 [1, 2, 2, 3, 4, 2, 5]))
    putStrLn (show (count3 2 [1, 2, 2, 3, 4, 2, 5]))
    putStrLn (show (count4 2 [1, 2, 2, 3, 4, 2, 5]))
