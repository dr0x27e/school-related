import Test.Hspec
import Test.QuickCheck

-- The functions to test
mhead1 :: [a] -> a
mhead1 (x:xs) = x
mhead1 [] = error "Empty list"

mhead2 :: [a] -> Maybe a
mhead2 = foldr (\x _ -> Just x) Nothing

-- Test suite
main :: IO ()
main = hspec $ do
  describe "mhead1" $ do
    it "returns the first element of a non-empty list" $ do
      mhead1 [1, 2, 3] `shouldBe` 1
      mhead1 ["a", "b", "c"] `shouldBe` "a"

  describe "mhead2" $ do
    it "returns Just the first element of a non-empty list" $ do
      mhead2 [1, 2, 3] `shouldBe` Just 1
      mhead2 ["a", "b", "c"] `shouldBe` Just "a"
    it "returns Nothing for an empty list" $ do
      mhead2 ([] :: [Int]) `shouldBe` Nothing
  
