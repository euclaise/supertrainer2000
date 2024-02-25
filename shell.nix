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
		python310Packages.uv
		stdenv.cc.cc.lib
		ninja
		cudaPackages_12_1.cudatoolkit linuxPackages.nvidia_x11
		snappy
	];
	shellHook = ''
	export CUDA_PATH=${pkgs.cudaPackages_12_1.cudatoolkit}
	export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
		pkgs.stdenv.cc.cc
		pkgs.linuxPackages.nvidia_x11
		pkgs.snappy
	]}
	export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
	export EXTRA_CCFLAGS="-I/usr/include"
	'';
}
