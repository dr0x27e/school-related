-- Creating an function that takes in "ANY" num, and returns their addition.
addNumber :: Num x => x -> x -> x
addNumber x y = x + y

-- We make a new type called Age that derives from Show so that we can turn it into string
-- And Read so that we can use read function on a string and get an Age.
-- Add Num and now we can use it with other Nums, so we can use '+' operator.
newtype Age = Age Int deriving (Show, Read, Num)
addAge :: Age -> Age -> Age
addAge x y = x + y

main :: IO ()
main = do
   putStrLn "What is you name?"
   name <- getLine

   putStrLn "And what is your age?"
   age <- getLine
   putStrLn ("Hello " ++ name ++ " In 10 years you will be " ++ show (addAge (read $ "Age " ++ age) 10))


