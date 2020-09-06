extern crate web_view;
extern crate rust_embed;
extern crate actix_web;
extern crate actix_rt;
extern crate mime_guess;

use web_view::*;
use rust_embed::RustEmbed;
use actix_web::body::Body;
use actix_web::App;
use actix_web::HttpResponse;
use actix_web::HttpRequest;
use actix_web::HttpServer;
use actix_web::web;
use std::borrow::Cow;
use std::thread;
use futures::executor::block_on;

#[derive(RustEmbed)]
#[folder = "../ui/crate/pkg"]
struct Asset;

fn assets(req: HttpRequest) -> HttpResponse {
    let path = if req.path() == "/" {
        "index.html"
    } else {
        // trim leading '/'
        &req.path()[1..]
    };

    match Asset::get(path) {
        Some(content) => {
            let body: Body = match content {
                Cow::Borrowed(bytes) => bytes.into(),
                Cow::Owned(bytes) => bytes.into(),
            };
            HttpResponse::Ok()
                .content_type(mime_guess::from_path(path).first_or_octet_stream().as_ref())
                .body(body)
        }
        None => HttpResponse::NotFound().body("404 Not Found"),
    }
}

fn main() {
    let (server_tx, server_rx) = std::sync::mpsc::channel();
    let (port_tx, port_rx) = std::sync::mpsc::channel();

    thread::spawn(move || {
        let sys = actix_rt::System::new("synth-server");

        let server = HttpServer::new(|| App::new().route("*", web::get().to(assets)))
            .bind("127.0.0.1:0")
            .unwrap();

        let port = server.addrs().first().unwrap().port();
        let server = server.run();

        port_tx.send(port).unwrap();
        server_tx.send(server).unwrap();
        sys.run().unwrap();
    });

    let port = port_rx.recv().unwrap();
    let server = server_rx.recv().unwrap();

    web_view::builder()
        .title("Synth")
        .content(Content::Url(format!("http://127.0.0.1:{}", port)))
        .size(800, 600)
        .resizable(true)
        .debug(true)
        .user_data(())
        .invoke_handler(invoke_handler)
        .run()
        .unwrap();
    
    block_on(server.stop(true));
}

fn invoke_handler(wv: &mut WebView<()>, arg: &str) -> WVResult {
    Ok(())
}
