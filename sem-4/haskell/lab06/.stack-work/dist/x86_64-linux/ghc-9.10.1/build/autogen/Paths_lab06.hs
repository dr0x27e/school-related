{-# LANGUAGE CPP #-}
{-# LANGUAGE NoRebindableSyntax #-}
#if __GLASGOW_HASKELL__ >= 810
{-# OPTIONS_GHC -Wno-prepositive-qualified-module #-}
#endif
{-# OPTIONS_GHC -fno-warn-missing-import-lists #-}
{-# OPTIONS_GHC -w #-}
module Paths_lab06 (
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
bindir     = "/home/thar/DrTerror/school-related/sem-4/haskell/lab06/.stack-work/install/x86_64-linux/642f34becf2a0adb5370481e7d69bf0b68dd6de06cfe815d858df5bb859ddfb4/9.10.1/bin"
libdir     = "/home/thar/DrTerror/school-related/sem-4/haskell/lab06/.stack-work/install/x86_64-linux/642f34becf2a0adb5370481e7d69bf0b68dd6de06cfe815d858df5bb859ddfb4/9.10.1/lib/x86_64-linux-ghc-9.10.1-69c3/lab06-0.1.0.0-EIPljO8b1GbDPLT5p0RjgY"
dynlibdir  = "/home/thar/DrTerror/school-related/sem-4/haskell/lab06/.stack-work/install/x86_64-linux/642f34becf2a0adb5370481e7d69bf0b68dd6de06cfe815d858df5bb859ddfb4/9.10.1/lib/x86_64-linux-ghc-9.10.1-69c3"
datadir    = "/home/thar/DrTerror/school-related/sem-4/haskell/lab06/.stack-work/install/x86_64-linux/642f34becf2a0adb5370481e7d69bf0b68dd6de06cfe815d858df5bb859ddfb4/9.10.1/share/x86_64-linux-ghc-9.10.1-69c3/lab06-0.1.0.0"
libexecdir = "/home/thar/DrTerror/school-related/sem-4/haskell/lab06/.stack-work/install/x86_64-linux/642f34becf2a0adb5370481e7d69bf0b68dd6de06cfe815d858df5bb859ddfb4/9.10.1/libexec/x86_64-linux-ghc-9.10.1-69c3/lab06-0.1.0.0"
sysconfdir = "/home/thar/DrTerror/school-related/sem-4/haskell/lab06/.stack-work/install/x86_64-linux/642f34becf2a0adb5370481e7d69bf0b68dd6de06cfe815d858df5bb859ddfb4/9.10.1/etc"

getBinDir     = catchIO (getEnv "lab06_bindir")     (\_ -> return bindir)
getLibDir     = catchIO (getEnv "lab06_libdir")     (\_ -> return libdir)
getDynLibDir  = catchIO (getEnv "lab06_dynlibdir")  (\_ -> return dynlibdir)
getDataDir    = catchIO (getEnv "lab06_datadir")    (\_ -> return datadir)
getLibexecDir = catchIO (getEnv "lab06_libexecdir") (\_ -> return libexecdir)
getSysconfDir = catchIO (getEnv "lab06_sysconfdir") (\_ -> return sysconfdir)



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
