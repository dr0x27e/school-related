import Test.Hspec
import Test.QuickCheck

-- Recursively:
mfact1 :: Int -> Int
mfact1 0 = 1
mfact1 x = x * mfact1 (x - 1)

-- Non-recursive:
mfact2 :: Int -> Int
mfact2 x = foldr (*) x [2..(x-1)]

main :: IO ()
main = hspec $ do
  describe "mfact1 (Recursive Factorial)" $ do
    it "computes factorial of 0" $
      mfact1 0 `shouldBe` 1

    it "computes factorial of 1" $
      mfact1 1 `shouldBe` 1

    it "computes factorial of 5" $
      mfact1 5 `shouldBe` 120

    it "computes factorial of 7" $
      mfact1 7 `shouldBe` 5040

  describe "mfact2 (Non-Recursive Factorial)" $ do
    it "computes factorial of 1" $
      mfact2 1 `shouldBe` 1

    it "computes factorial of 5" $
      mfact2 5 `shouldBe` 120

    it "computes factorial of 7" $
      mfact2 7 `shouldBe` 5040
