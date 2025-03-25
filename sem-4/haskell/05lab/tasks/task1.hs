-- Using patternmatching:
mhead1 :: [a] -> a
mhead1 (x:_) = x
mhead1 [] = error "empty list"

-- Using Function guards:
mhead2 :: [a] -> a
mhead2 xs
	| null xs = error "empty list"
	| otherwise = xs !! 0

-- Using if else:
mhead3 :: [a] -> a
mhead3 xs = if null xs then error "empty lsit" else xs !! 0

-- Using let in:
mhead4 :: [a] -> a
mhead4 xs = 
	let err = error "empty list"
	in if null xs then err else xs !! 0

-- Using where:
mhead5 :: [a] -> a
mhead5 xs
	| null xs = error "empty list"
	| otherwise = helperHead xs
	where
	    helperHead :: [a] -> a
	    helperHead ys = ys !! 0


-- Using case of:
mhead6 :: [a] -> a
mhead6 xs = case xs of
	[] -> error "empty list"
	(x:_) -> x

main = do
    putStrLn ( show (mhead1 [1, 2, 3]) )
    putStrLn ( show (mhead2 [2, 2, 1]) )
    putStrLn ( show (mhead3 [3, 3, 1]) )
    putStrLn ( show (mhead4 [4, 3, 1]) )
    putStrLn ( show (mhead5 [5, 3, 1]) )
    putStrLn ( show (mhead6 [6, 4, 7]) )
