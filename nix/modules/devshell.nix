{ inputs, ... }:
{
  perSystem = { system, ... }:
    let
      pkgs = import inputs.nixpkgs {
        inherit system;
        overlays = [ inputs.rust-overlay.overlays.default ];
      };
    in {
    devShells.default = pkgs.mkShell {
      name = "yolo-rerun-demo-shell";

      packages = with pkgs; [
        cargo
        clang
        ffmpeg
        llvmPackages.libclang
        openssl
        pkg-config
        rerun
        libx11
        libxcursor
        libxi
        libxinerama
        libxrandr
        libxxf86vm
        (rust-bin.stable."1.96.0".default)
      ];

      LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.libx11
        pkgs.libxcursor
        pkgs.libxi
        pkgs.libxinerama
        pkgs.libxrandr
        pkgs.libxxf86vm
      ];
    };

    };
}
