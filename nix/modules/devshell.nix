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

    packages.default = pkgs.rustPlatform.buildRustPackage {
      pname = "yolo-rerun-demo";
      version = "0.1.0";
      src = ../..;
      cargoLock.lockFile = ../../Cargo.lock;

      cargoBuildFeatures = [ "raw-viewer" ];
      doCheck = false;

      nativeBuildInputs = with pkgs; [
        clang
        llvmPackages.libclang
        makeWrapper
        pkg-config
      ];
      buildInputs = with pkgs; [
        ffmpeg
        onnxruntime
        openssl
        rerun
        libx11
        libxcursor
        libxi
        libxinerama
        libxrandr
        libxxf86vm
      ];

      LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
      ORT_LIB_LOCATION = "${pkgs.onnxruntime}/lib";
      ORT_PREFER_DYNAMIC_LINK = "1";
      YOLO_DEFAULT_MODEL = "${placeholder "out"}/share/yolo-rerun-demo/yolo26n.onnx";

      postInstall = ''
        install -Dm644 yolo26n.onnx "$out/share/yolo-rerun-demo/yolo26n.onnx"
      '';

      postFixup = ''
        wrapProgram "$out/bin/yolo-rerun-demo" \
          --prefix PATH : "${pkgs.lib.makeBinPath [ pkgs.rerun ]}" \
          --prefix LD_LIBRARY_PATH : "${pkgs.lib.makeLibraryPath [
            pkgs.ffmpeg
            pkgs.onnxruntime
            pkgs.openssl
            pkgs.libx11
            pkgs.libxcursor
            pkgs.libxi
            pkgs.libxinerama
            pkgs.libxrandr
            pkgs.libxxf86vm
          ]}"
      '';

      meta = {
        description = "YOLO webcam inference with Rerun visualization";
        mainProgram = "yolo-rerun-demo";
        platforms = pkgs.lib.platforms.linux;
      };
    };
    };
}
