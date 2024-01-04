{
	pkgs ? import <nixpkgs> { config.allowUnfree = true; },
}:
 pkgs.mkShell {
	buildInputs = with pkgs; [ python310 poetry stdenv.cc.cc.lib ];
	shellHook = ''
	export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
		pkgs.stdenv.cc.cc
	]}

	exec poetry shell
	'';
}
