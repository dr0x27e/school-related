-- Creating an function that takes in "ANY" num, and returns their addition.
addNumber :: Num x => x -> x -> x
addNumber x y = x + y

-- We make a new type called Age that derives from Show so that we can turn it into string
-- And Read so that we can use read function on a string and get an Age.
newtype Age = Age Int deriving (Show, Read)
addAge :: Age -> Age -> Age
addAge (Age x) (Age y) = Age (x + y)

main :: IO ()
main = do
   putStrLn "What is you name?"
   name <- getLine

   putStrLn "And what is your age?"
   age <- getLine
   putStrLn ("Hello " ++ name ++ " In 10 years you will be " ++ show (addAge (read $ "Age " ++ age :: Age) (Age 10)))


