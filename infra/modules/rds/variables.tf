variable "app_name"               { type = string }
variable "environment"            { type = string }
variable "db_instance_class"      { type = string }
variable "db_security_group_id"   { type = string }
variable "subnet_ids"             { type = list(string) }
variable "db_password_secret_arn" { type = string }
