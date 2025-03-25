module Lib 
  ( countScore
  ) where

countScore :: String -> Int
countScore txt = sum (map processLine (lines txt))


-- | Process a single line of the input data.
-- Each line represents a game, the first three numbers are the "Winning Numbers", 
-- and the next five are the lottery numbers. Calculate the score for the single game.
--
-- >>> processLine "4 21 10 5 4 21 13 11"
-- 5
--
-- >>> processLine "4 21 10 5 4 21 13 11 10"
-- 7
--
-- >>> processLine "4 21 10 5 4 21 13 11 10 10"
-- 11
--
-- >>> processLine "10 21 10 5 4 21 13 11 10"
-- 8
--
-- >>> processLine "10 21 10 5 8 20 13 11"
-- 0
--
-- >>> processLine "10 10 10 5 4 21 13 11 10 10 10"
-- 56
--
-- >>> processLine "8 14 16 5 8 14 16 14"
-- 9
-- 
-- >>> processLine "8 14 16 5 8 18 16 12"
-- 3
--
-- >>> processLine "35 35 35 1 5 6 35 16"
-- 32
--
-- >>> processLine "35 35 6 1 5 6 35 35"
-- 49
-- 
-- >>> processLine "35 35 35 1 5 6 35 35"
-- 96
-- 
processLine :: String -> Int
processLine msg = sum $
  (\(fr, res) ->
    let frst = format fr
    in snd (foldl (\(frAcc, resAcc) x ->
        let (newFr, newVal) = updateValue x frAcc
        in (newFr, resAcc ++ [newVal])
      ) (frst, []) res)
  ) (splitAt 3 (read <$> words msg))

-- Format the first three elements into (num, power-based value)
format :: [Int] -> [(Int, Int)]
format fr = (\(x, y) -> (x, 2 ^ (x `div` 10) * 2 ^ (y - 1))) <$> (foldr occ [] fr)

-- Count occurrences and store them in a list of tuples
occ :: Int -> [(Int, Int)] -> [(Int, Int)]
occ x [] = [(x, 1)]
occ x ((y, count):xs) = if x == y then (y, count + 1) : xs else (y, count) : occ x xs

-- Update the value of x in frAcc and modify the replacement value in res
updateValue :: Int -> [(Int, Int)] -> ([(Int, Int)], Int)
updateValue x fr =
  case lookup x fr of
    Just val -> let updatedFr = map (\(k, v) -> if k == x then (k, v * 2) else (k, v)) fr
                in (updatedFr, val)
    Nothing  -> (fr, 0)

-- Counting number of occurrances
countOcc :: [Int] -> [Int] -> [Int]
countOcc l1 [] = []
countOcc l1 (x:xs) = length (filter (== x) l1) : countOcc l1 xs

