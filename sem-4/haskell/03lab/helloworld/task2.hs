main :: IO ()
main = do
   putStrLn "What is you name?"
   name <- getLine
   putStrLn $ "Hello " ++ name
