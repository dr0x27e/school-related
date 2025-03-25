module Lib
    ( decodeMessage,
    decodeMessageImproved
    ) where


-- | Decode an intergalactic message from a string.
-- The message is a sequence of integers separated by spaces.
--
-- >>> decodeMessage "5 5 5 8 1 2 3 4 9 8 2 3 4 1"
-- Nothing
--
-- >>> decodeMessage "5 5 5 8 1 2 3 4 9 8 2 3 4 9"
-- Nothing
--
-- >>> decodeMessage "5 5 5 8 1 2 3 4 9 8 2 3 4 8"
-- Just 3
-- 
-- >>> decodeMessage "5 5 5 1 2 3 4 8 2 3"
-- Nothing
decodeMessage :: String -> Maybe Int
decodeMessage msg = 
    let lis = read <$> (words msg)
    in minmaxMaybe lis >>= isEvenMaybe >>= (\n -> Just (length (filter (== n) lis)))


minmaxMaybe :: [Int] -> Maybe (Int, Int)
minmaxMaybe xs =
    let (minN, seen1, maxN, seen2) = foldl check (maxBound, 0, minBound, 0) xs
    in  if seen1 == 1 && seen2 == 1
        then Just (minN, maxN)
        else Nothing
  where
    check (minN, seen1, maxN, seen2) x
      | x < minN  = (x, 1, maxN, seen2)
      | x == minN = (minN, seen1 + 1, maxN, seen2)
      | x > maxN  = (minN, seen1, x, 1)
      | x == maxN = (minN, seen1, maxN, seen2 + 1)
      | otherwise = (minN, seen1, maxN, seen2)

isEvenMaybe :: (Int, Int) -> Maybe Int
isEvenMaybe (minN, maxN)
    | even (minN + maxN) = Just ((minN + maxN) `div` 2)
    | otherwise          = Nothing

-- | Decode an intergalactic message from a string.
-- The message is a sequence of integers separated by spaces.
-- This is an improved version of the previous function, with a more
-- informative error messages.
--
-- >>> decodeMessageImproved "5 5 5 8 1 2 3 4 9 8 2 3 4 1"
-- Left "Communication interference detected: minimum number not Unique"
--
-- >>> decodeMessageImproved "5 5 5 8 1 2 3 4 9 8 2 3 4 9"
-- Left "Communication interference detected: maximum number not Unique"
--
-- >>> decodeMessageImproved "5 5 5 8 1 2 3 4 9 8 2 3 4 8"
-- Right 3
--
-- >>> decodeMessageImproved "5 5 5 1 2 3 4 8 2 3"
-- Left "Communication interference detected: midPoint not even"
decodeMessageImproved :: String -> Either String Int
decodeMessageImproved msg =
    let lis = read <$> words msg
    in minmaxEither lis >>= isEvenEither >>= (\n -> Right (length (filter (== n) lis)))

minmaxEither :: [Int] -> Either String (Int, Int)
minmaxEither xs =
    let (minN, seen1, maxN, seen2) = foldl check (maxBound, 0, minBound, 0) xs
    in  if seen1 /= 1 then      Left "Communication interference detected: minimum number not Unique" 
        else if seen2 /= 1 then Left "Communication interference detected: maximum number not Unique"
	else                    Right (minN, maxN)
  where
    check (minN, seen1, maxN, seen2) x
      | x < minN  = (x, 1, maxN, seen2)
      | x == minN = (minN, seen1 + 1, maxN, seen2)
      | x > maxN  = (minN, seen1, x, 1)
      | x == maxN = (minN, seen1, maxN, seen2 + 1)
      | otherwise = (minN, seen1, maxN, seen2)

isEvenEither :: (Int, Int) -> Either String Int
isEvenEither (minN, maxN)
    | even (minN + maxN) = Right ((minN + maxN) `div` 2)
    | otherwise          = Left "Communication interference detected: midPoint not even"

