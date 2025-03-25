
-- TASK 1
mreverse :: [a] -> [a] 			-- mreverse takes in list of "any" and returns the same "any"
mreverse [] = [] 			-- Base case to stop recursion.
mreverse (x:xs) = mreverse xs ++ [x]	-- Pattern match and do mreverse on rest and concat with first element


-- TASK 2
{-
 - Padding function for mulTable:
 - Takes in an Int returns a string.
 - First you calculate how big the number is.
 - This is done by using show which takes Int and returns it as a string,
 - then the length of this is taken and subtracted from 3 so that we will,
 - always atleast have 1 space between numbers, then we concat all these ' ',
 - and the number.
-}
pad :: Int -> String
pad n = replicate (3 - length (show n)) ' ' ++ show n


{-
 - mulTable takes in Int returns nothing "()"
 - With this int it does mapM_ which is map but instead of returning all the elements from,
 - an array back as an array, it returns them 1 after another. we map all the outputs from mapM_,
 - with a composition of putStrLn and a helper function called formatRow,
 - the putStrLn is self explanitory what does.
 - FormatRow takes an Int and returns a string, we first map an array [1..size] with * i which multiplies,
 - Each element in the array with i. Then this output is sent to pad. finally size the map returns the array,
 - we need to flatten this array of now strings, we do this with unwords.
-}
mulTable :: Int -> IO ()
mulTable size = mapM_ (putStrLn . formatRow) [1..size]
    where
        formatRow :: Int -> String
        formatRow i = unwords (map (pad . (* i)) [1..size])
	

-- TASK 3
{-
 - FindMaxValueAndCount -> FindMax
 - FindMax takes in an array of int's prepared by other functions.
 - To make the function go faster we use something called accumalators:
 - foldl is a inbuilt function that reduces an array into a single value:
 - foldl takes 2 parameters 1: binary function and 2: an initial accumalator.
 - An accumalator is a "parameter" that does not need to be called from memory,
 - This makes it so that for every element in the array we can just use the same,
 - value saving time.
 - The Binary function is a helper function called check, it checks if the current maxVal,
 - is bigger, equal or less than the current element in the array.
 - Case Bigger: New biggest val found, reset accumalator with new maxVal and count of 1.
 - Case Same: Same size maxVal found, increment counter.
 - Case Smaller: Smaller value found, ignore and continue.
-}
findMaxValueAndCount :: [Int] -> (Int, Int)
findMaxValueAndCount = foldl check (minBound, 0)
    where
        check (maxVal, count) x
	    | x > maxVal = (x, 1) 		  -- New Max Found. reset maxVal and count.
	    | x == maxVal = (maxVal, count + 1)   -- Same Max Found, increment the counter.
	    | otherwise = (maxVal, count)	  -- Smaller value found, Ignore the value.




main = do
    -- TASK 1
    print (mreverse [1, 2, 3])
    
    -- TASK 2
    mulTable 10
    
    -- TASK 3
    contents <- getContents -- Fetch IO output from getContents.
    {-
     - function pipeline:
     - first split the string from contents into an array of strings/words
     - Take every third element (which is an int.)
     - map read function on this array (read takes string "42" and turns into int 42)
     - send this to findMaxValueAndCount.
    -}
    let (maxVal, count) = findMaxValueAndCount $ map read . everyThird . words $ contents
    putStrLn ("Max : " ++ show maxVal ++ ", Count : " ++ show count)


-- Helper function for task 3
{-
 - everyThird takes an array of strings and returns an array of strings.
 - It takes every third element and then constructs an array with these integers.
-}
everyThird :: [String] -> [String]
everyThird (_:_:n:ns) = n : everyThird ns
everyThird _ = []				-- Incase there is no more/uneven number elements.
