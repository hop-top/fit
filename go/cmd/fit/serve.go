package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"hop.top/fit"
	"hop.top/kit/cli"
)

func serveCmd(_ *cli.Root) *cobra.Command {
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

			advisor := &stubAdvisor{model: advisorModel}

			mux := http.NewServeMux()
			mux.HandleFunc("POST /advise", handleAdvise(advisor, timeout))

			srv := &http.Server{
				Addr:         addr,
				Handler:      mux,
				ReadTimeout:  time.Duration(timeout) * time.Millisecond,
				WriteTimeout: time.Duration(timeout*2) * time.Millisecond,
			}

			errCh := make(chan error, 1)
			go func() {
				fmt.Fprintf(cmd.OutOrStdout(), "fit advisor serving on %s\n", addr)
				errCh <- srv.ListenAndServe()
			}()

			sigCh := make(chan os.Signal, 1)
			signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

			select {
			case err := <-errCh:
				return err
			case sig := <-sigCh:
				fmt.Fprintf(cmd.OutOrStdout(), "\nreceived %s, shutting down\n", sig)
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				return srv.Shutdown(ctx)
			}
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
