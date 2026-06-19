use actix_web::http::StatusCode;
use actix_web::{HttpResponse, ResponseError, body::BoxBody};
use serde::Serialize;
use serde_json::to_string_pretty;
use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub status: u16,
}

impl Display for ErrorResponse {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", to_string_pretty(self).unwrap())
    }
}

impl ResponseError for ErrorResponse {
    fn status_code(&self) -> StatusCode {
        StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
    }

    fn error_response(&self) -> HttpResponse<BoxBody> {
        HttpResponse::build(self.status_code())
            .json(serde_json::json!({"error": self.error.clone()}))
    }
}

impl From<actix_web::Error> for ErrorResponse {
    fn from(err: actix_web::Error) -> Self {
        ErrorResponse {
            error: err.to_string(),
            status: err.as_response_error().status_code().as_u16(),
        }
    }
}
