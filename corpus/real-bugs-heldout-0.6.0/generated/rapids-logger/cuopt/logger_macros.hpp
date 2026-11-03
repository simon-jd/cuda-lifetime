/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rapids_logger/log_levels.h>

// Default to info level if not specified.
#if !defined(CUOPT_LOG_ACTIVE_LEVEL)
#define CUOPT_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_INFO
#endif

// Macros for easier logging, similar to spdlog.
#define CUOPT_LOGGER_CALL(logger, level, ...) (logger).log(level, __VA_ARGS__)

#if CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_TRACE
#define CUOPT_LOG_TRACE(...) \
  CUOPT_LOGGER_CALL(cuopt::default_logger(), rapids_logger::level_enum::trace, __VA_ARGS__)
#else
#define CUOPT_LOG_TRACE(...) (void)0
#endif

#if CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_DEBUG
#define CUOPT_LOG_DEBUG(...) \
  CUOPT_LOGGER_CALL(cuopt::default_logger(), rapids_logger::level_enum::debug, __VA_ARGS__)
#else
#define CUOPT_LOG_DEBUG(...) (void)0
#endif

#if CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_INFO
#define CUOPT_LOG_INFO(...) CUOPT_LOGGER_CALL(cuopt::default_logger(), rapids_logger::level_enum::info, __VA_ARGS__)
#else
#define CUOPT_LOG_INFO(...) (void)0
#endif

#if CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_WARN
#define CUOPT_LOG_WARN(...) CUOPT_LOGGER_CALL(cuopt::default_logger(), rapids_logger::level_enum::warn, __VA_ARGS__)
#else
#define CUOPT_LOG_WARN(...) (void)0
#endif

#if CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_ERROR
#define CUOPT_LOG_ERROR(...) \
  CUOPT_LOGGER_CALL(cuopt::default_logger(), rapids_logger::level_enum::error, __VA_ARGS__)
#else
#define CUOPT_LOG_ERROR(...) (void)0
#endif

#if CUOPT_LOG_ACTIVE_LEVEL <= RAPIDS_LOGGER_LOG_LEVEL_CRITICAL
#define CUOPT_LOG_CRITICAL(...) \
  CUOPT_LOGGER_CALL(cuopt::default_logger(), rapids_logger::level_enum::critical, __VA_ARGS__)
#else
#define CUOPT_LOG_CRITICAL(...) (void)0
#endif
