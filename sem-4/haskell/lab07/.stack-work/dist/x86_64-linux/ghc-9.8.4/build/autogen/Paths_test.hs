{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
#if __GLASGOW_HASKELL__ >= 810
{-# OPTIONS_GHC -Wno-prepositive-qualified-module #-}
#endif
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -w #-}
module Paths_test (
    version,
    getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir,
    getDataFileName, getSysconfDir
  ) where


import qualified Control.Exception as Exception
import qualified Data.List as List
import Data.Version (Version(..))
import System.Environment (getEnv)
import Prelude


#if defined(VERSION_base)

#if MIN_VERSION_base(4,0,0)
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#else
catchIO :: IO a -> (Exception.Exception -> IO a) -> IO a
#endif

#else
catchIO :: IO a -> (Exception.IOException -> IO a) -> IO a
#endif
catchIO = Exception.catch

version :: Version
version = Version [0,1,0,0] []

getDataFileName :: FilePath -> IO FilePath
getDataFileName name = do
  dir <- getDataDir
  return (dir `joinFileName` name)

getBinDir, getLibDir, getDynLibDir, getDataDir, getLibexecDir, getSysconfDir :: IO FilePath




bindir, libdir, dynlibdir, datadir, libexecdir, sysconfdir :: FilePath
bindir     = "/home/thar/DrTerror/school-related/sem-4/haskell/lab07/.stack-work/install/x86_64-linux/ed75ed6cffd5ddfeb54e71f3ee6f57a1927b9be505d16c66e0d90bc57160f5f9/9.8.4/bin"
libdir     = "/home/thar/DrTerror/school-related/sem-4/haskell/lab07/.stack-work/install/x86_64-linux/ed75ed6cffd5ddfeb54e71f3ee6f57a1927b9be505d16c66e0d90bc57160f5f9/9.8.4/lib/x86_64-linux-ghc-9.8.4/test-0.1.0.0-KzwGtagyM8UD2BzJXtW7N7"
dynlibdir  = "/home/thar/DrTerror/school-related/sem-4/haskell/lab07/.stack-work/install/x86_64-linux/ed75ed6cffd5ddfeb54e71f3ee6f57a1927b9be505d16c66e0d90bc57160f5f9/9.8.4/lib/x86_64-linux-ghc-9.8.4"
datadir    = "/home/thar/DrTerror/school-related/sem-4/haskell/lab07/.stack-work/install/x86_64-linux/ed75ed6cffd5ddfeb54e71f3ee6f57a1927b9be505d16c66e0d90bc57160f5f9/9.8.4/share/x86_64-linux-ghc-9.8.4/test-0.1.0.0"
libexecdir = "/home/thar/DrTerror/school-related/sem-4/haskell/lab07/.stack-work/install/x86_64-linux/ed75ed6cffd5ddfeb54e71f3ee6f57a1927b9be505d16c66e0d90bc57160f5f9/9.8.4/libexec/x86_64-linux-ghc-9.8.4/test-0.1.0.0"
sysconfdir = "/home/thar/DrTerror/school-related/sem-4/haskell/lab07/.stack-work/install/x86_64-linux/ed75ed6cffd5ddfeb54e71f3ee6f57a1927b9be505d16c66e0d90bc57160f5f9/9.8.4/etc"

getBinDir     = catchIO (getEnv "test_bindir")     (\_ -> return bindir)
getLibDir     = catchIO (getEnv "test_libdir")     (\_ -> return libdir)
getDynLibDir  = catchIO (getEnv "test_dynlibdir")  (\_ -> return dynlibdir)
getDataDir    = catchIO (getEnv "test_datadir")    (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "test_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "test_sysconfdir") (\_ -> return sysconfdir)



joinFileName :: String -> String -> FilePath
joinFileName ""  fname = fname
joinFileName "." fname = fname
joinFileName dir ""    = dir
joinFileName dir fname
  | isPathSeparator (List.last dir) = dir ++ fname
  | otherwise                       = dir ++ pathSeparator : fname

pathSeparator :: Char
pathSeparator = '/'

isPathSeparator :: Char -> Bool
isPathSeparator c = c == '/'
