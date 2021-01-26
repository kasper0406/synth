const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
// const webpack = require('webpack');
const WasmPackPlugin = require("@wasm-tool/wasm-pack-plugin");

module.exports = {
    entry: './main.js',
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: 'main.js',
        library: 'synth',
        libraryTarget: 'window',
    },
    devtool: 'inline-source-map',
    devServer: {
        contentBase: './dist'
    },
    plugins: [
        new HtmlWebpackPlugin({
            title: 'TubeSynth',
        }),
        new WasmPackPlugin({
            crateDirectory: path.resolve(__dirname, "minimal")
        })
    ],
    mode: 'development',
    experiments: {
        asyncWebAssembly: true
    }
};
