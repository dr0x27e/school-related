mhead1 :: [a] -> a
mhead1 (x:xs) = x
mhead [] = []
-- Using pattern matching
-- Split the list into first element and rest of the elements.

-- foldr is a higher level funciton that goes through a list from the right to the left
-- and returns a single value.
-- we use foldr instead of foldl as we want to end on the start of the list.
-- fold takes 2 parameters, 1: function to use on all values in the list and an accumalator
-- An accumalator is a value that does not need to be fetched from memory every time its called.
-- (\x _ -> Just x) Nothing <- is a lambda function \x is the current element, _ is the accumalator,
-- Nothing is the initial state of the accumalator. We begin from the right:
-- [1, 2, 3] -> (3, _) Just 3 then (2, _) Just 2 then finally (1, _) Just 1 and this is our final output.
mhead2 :: [a] -> Maybe a
mhead2 = foldr (\x _ -> Just x) Nothing
mhead2 [] = []

main :: IO ()
main = do
    putStrLn "hello"
