package = "autobw"
version = "scm-1"

source = {
   url = "git://github.com/bshillingford/autobw.git"
}

description = {
   summary = "Automatically perform backwards passes",
   detailed = [[
       Automatically induces the backwards pass based on a sequence of forward() calls 
       on nn Modules and Criterions, without needing to compute them manually.
   ]],
   homepage = "https://github.com/bshillingford/autobw",
   license = "BSD"
}

dependencies = {
   "lua >= 5.1",
}

build = {
   type = "builtin",
   modules = {
      ["autobw.init"] = "init.lua",
      ["autobw.tape"] = "tape.lua",
      ["autobw.utils"] = "utils.lua",
   }
}
