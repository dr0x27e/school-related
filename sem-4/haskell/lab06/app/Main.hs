module Main (main) where

import Lib

main :: IO ()
main = do
    msg <- getContents
    case decodeMessage msg of
        Just n -> putStrLn $ "The message is " ++ show n ++ " characters long."
        Nothing -> putStrLn "The message cannot be decoded. Interference detected."
    
    case decodeMessageImproved msg of
        Right n -> print n
	Left n -> putStrLn n
