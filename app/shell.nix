{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  packages = [ pkgs.nodejs_latest pkgs.pnpm ];

  shellHook = ''
    pnpm i
  '';
}