variable "app_name"              { type = string }
variable "environment"           { type = string }
variable "vpc_id"                { type = string }
variable "public_subnet_ids"     { type = list(string) }
variable "alb_security_group_id" { type = string }
variable "enable_https"          { type = bool;   default = false }
variable "acm_certificate_arn"   { type = string; default = "" }
