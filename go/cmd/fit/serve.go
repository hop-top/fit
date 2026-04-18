package main

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/spf13/cobra"
	"hop.top/fit"
	"hop.top/kit/api"
	"hop.top/kit/cli"
	"hop.top/kit/log"
)

func serveCmd(root *cli.Root) *cobra.Command {
	var (
		addr         string
		advisorModel string
		timeout      int
	)

	cfg := Config()

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start advisor HTTP server",
		Long: `Start an HTTP server that exposes the advisor API.

The server accepts POST /advise requests with context JSON and returns
advice conforming to advice-format-v1.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			if !cmd.Flags().Changed("addr") {
				addr = cfg.Addr
			}
			if !cmd.Flags().Changed("model") {
				advisorModel = cfg.Model
			}
			if !cmd.Flags().Changed("timeout") {
				timeout = cfg.Timeout
			}

			logger := log.New(root.Viper)
			advisor := &stubAdvisor{model: advisorModel}

			router := api.NewRouter(
				api.WithMiddleware(
					api.Recovery(func(v any, r *http.Request) {
						logger.Error("panic recovered", "err", v, "path", r.URL.Path)
					}),
					api.Logger(func(msg string, args ...any) {
						logger.Info(msg, args...)
					}),
					api.RequestID(),
				),
			)
			router.Handle("POST", "/advise", handleAdvise(advisor, timeout))

			fmt.Fprintf(cmd.OutOrStdout(), "fit advisor serving on %s\n", addr)
			if err := api.ListenAndServeWithSignals(addr, router,
				api.WithReadTimeout(time.Duration(timeout)*time.Millisecond),
				api.WithWriteTimeout(time.Duration(timeout*2)*time.Millisecond),
			); err != nil {
				if isAddrInUse(err) {
					return errPortInUse(addr, err)
				}
				return err
			}
			return nil
		},
	}

	cmd.Flags().StringVarP(&addr, "addr", "a", cfg.Addr, "listen address")
	cmd.Flags().StringVarP(&advisorModel, "model", "m", cfg.Model, "advisor model identifier")
	cmd.Flags().IntVarP(&timeout, "timeout", "t", cfg.Timeout, "request timeout in ms")

	return cmd
}

func handleAdvise(
	advisor fit.Advisor,
	timeoutMs int,
) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(timeoutMs)*time.Millisecond)
		defer cancel()

		var input map[string]any
		if err := jsonDecode(r.Body, &input); err != nil {
			http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
			return
		}

		advice, err := advisor.GenerateAdvice(ctx, input)
		if err != nil {
			http.Error(w, "advisor error: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := jsonEncode(w, advice); err != nil {
			http.Error(w, "encoding error: "+err.Error(), http.StatusInternalServerError)
			return
		}
	}
}

// stubAdvisor is a placeholder until real advisor implementations land.
type stubAdvisor struct{ model string }

func (a *stubAdvisor) GenerateAdvice(_ context.Context, input map[string]any) (*fit.Advice, error) {
	return &fit.Advice{
		Domain:       "generic",
		SteeringText: "No specialized advice available yet.",
		Confidence:   0.5,
		Version:      "1.0",
		Metadata:     input,
	}, nil
}

func (a *stubAdvisor) ModelID() string { return a.model }
