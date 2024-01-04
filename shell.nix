{
	pkgs ? import <nixpkgs> {
		config = {
			allowUnfree = true;
			cudaSupport = true;
		};
	},
}:
 pkgs.mkShell {
	buildInputs = with pkgs; [
		python310
		poetry
		stdenv.cc.cc.lib
		cudatoolkit linuxPackages.nvidia_x11
	];
	shellHook = ''
	export CUDA_PATH=${pkgs.cudatoolkit}
	export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
		pkgs.stdenv.cc.cc
		pkgs.linuxPackages.nvidia_x11
	]}
	export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
	export EXTRA_CCFLAGS="-I/usr/include"

	exec poetry shell
	'';
}
