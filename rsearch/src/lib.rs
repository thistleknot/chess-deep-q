//! rsearch — full-width alpha-beta + quiescence over the repo's 809-feature linear eval
//! (spec/rust-search.spec.md). Eval mirrors cem_loop.encode_features EXACTLY: 12x64 piece
//! planes + turn + 40 KnightCap-donor features, same normalizations, same indices.
//! Search: iterative-deepening negamax, full width, captures-until-quiet quiescence with
//! stand-pat, tanh at the leaf (values White-absolute in [-1,1] at the Python boundary).

use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves,
    BitBoard, Board, Color, File, Move, Piece, Rank, Square,
};
use pyo3::prelude::*;

const NFEAT: usize = 809;
const TYPE_VAL: [f64; 6] = [1.0, 3.0, 3.0, 5.0, 9.0, 0.0]; // P N B R Q K

fn bb_u64(bb: BitBoard) -> u64 {
    bb.0
}

fn file_mask(f: usize) -> u64 {
    0x0101_0101_0101_0101u64 << f
}

fn adj_files(f: usize) -> u64 {
    let mut m = 0u64;
    if f > 0 {
        m |= file_mask(f - 1);
    }
    if f < 7 {
        m |= file_mask(f + 1);
    }
    m
}

/// front span for passed-pawn test: same+adjacent files, ranks strictly ahead
fn front_span(white: bool, sq: usize) -> u64 {
    let f = sq % 8;
    let r = sq / 8;
    let files = file_mask(f) | adj_files(f);
    let ahead: u64 = if white {
        if r >= 7 {
            0
        } else {
            u64::MAX << (8 * (r + 1))
        }
    } else if r == 0 {
        0
    } else {
        (1u64 << (8 * r)) - 1
    };
    files & ahead
}

struct Eval {
    w: [f64; NFEAT],
    bias: f64,
}

impl Eval {
    /// White-absolute pre-tanh linear score; feature semantics == cem_loop.encode_features.
    fn score(&self, board: &Board) -> f64 {
        let occ_w = bb_u64(board.colors(Color::White));
        let occ_b = bb_u64(board.colors(Color::Black));
        let occ = [occ_w, occ_b];
        let occ_all = occ_w | occ_b;
        let pieces_bb = [
            bb_u64(board.pieces(Piece::Pawn)),
            bb_u64(board.pieces(Piece::Knight)),
            bb_u64(board.pieces(Piece::Bishop)),
            bb_u64(board.pieces(Piece::Rook)),
            bb_u64(board.pieces(Piece::Queen)),
            bb_u64(board.pieces(Piece::King)),
        ];
        let mut s = self.bias;
        // [0:768] planes: White P..K then Black P..K, feature = plane*64 + sq
        for side in 0..2 {
            for pt in 0..6 {
                let mut m = pieces_bb[pt] & occ[side];
                while m != 0 {
                    let sq = m.trailing_zeros() as usize;
                    s += self.w[(side * 6 + pt) * 64 + sq];
                    m &= m - 1;
                }
            }
        }
        // [768] side to move
        if board.side_to_move() == Color::White {
            s += self.w[768];
        }
        // attack pass: unions + per-piece (sq, type, mask); king in union, not in list
        let mut atk = [0u64, 0u64];
        let mut plist: [Vec<(usize, usize, u64)>; 2] = [Vec::with_capacity(16), Vec::with_capacity(16)];
        for side in 0..2 {
            let color = if side == 0 { Color::White } else { Color::Black };
            let blockers = BitBoard(occ_all);
            let mut a = 0u64;
            for pt in 0..5 {
                let mut m = pieces_bb[pt] & occ[side];
                while m != 0 {
                    let sqi = m.trailing_zeros() as usize;
                    let sq = Square::index(sqi);
                    let am = match pt {
                        0 => bb_u64(get_pawn_attacks(sq, color)),
                        1 => bb_u64(get_knight_moves(sq)),
                        2 => bb_u64(get_bishop_moves(sq, blockers)),
                        3 => bb_u64(get_rook_moves(sq, blockers)),
                        _ => bb_u64(get_bishop_moves(sq, blockers)) | bb_u64(get_rook_moves(sq, blockers)),
                    };
                    a |= am;
                    plist[side].push((sqi, pt, am));
                    m &= m - 1;
                }
            }
            let k = pieces_bb[5] & occ[side];
            if k != 0 {
                a |= bb_u64(get_king_moves(Square::index(k.trailing_zeros() as usize)));
            }
            atk[side] = a;
        }
        for side in 0..2 {
            let opp = 1 - side;
            let own = occ[side];
            let sign = if side == 0 { 1.0 } else { 1.0 }; // weights carry the sign per block
            let _ = sign;
            let own_p = pieces_bb[0] & own;
            let opp_p = pieces_bb[0] & occ[opp];
            let own_r = pieces_bb[3] & own;
            // [769:771] bishop pair
            if (pieces_bb[2] & own).count_ones() >= 2 {
                s += self.w[769 + side];
            }
            // hung
            let hung = own & atk[opp] & !atk[side] & !pieces_bb[5];
            s += self.w[787 + side] * (hung.count_ones() as f64) / 5.0;
            // one pass: mobility, safe mobility, hung value, rook file features
            let mut mob = [0u32; 4];
            let mut smob = [0u32; 4];
            let mut hval = 0.0;
            let mut ropen = 0u32;
            let mut rhalf = 0u32;
            let mut connected = 0.0;
            for &(sqi, pt, am) in &plist[side] {
                if pt != 0 {
                    let free = am & !own;
                    mob[pt - 1] += free.count_ones();
                    smob[pt - 1] += (free & !atk[opp]).count_ones();
                    if pt == 3 {
                        let fmask = file_mask(sqi % 8);
                        if pieces_bb[0] & fmask == 0 {
                            ropen += 1;
                        } else if own_p & fmask == 0 {
                            rhalf += 1;
                        }
                        if am & own_r & !(1u64 << sqi) != 0 {
                            connected = 1.0;
                        }
                    }
                }
                if hung != 0 && (1u64 << sqi) & hung != 0 {
                    hval += TYPE_VAL[pt];
                }
            }
            for ti in 0..4 {
                s += self.w[771 + 4 * side + ti] * (mob[ti] as f64) / 14.0;
                s += self.w[779 + 4 * side + ti] * (smob[ti] as f64) / 14.0;
            }
            s += self.w[789 + side] * hval / 9.0;
            s += self.w[803 + side] * (ropen as f64) / 2.0;
            s += self.w[805 + side] * (rhalf as f64) / 2.0;
            s += self.w[807 + side] * connected;
            // king ring
            let k = pieces_bb[5] & own;
            if k != 0 {
                let ring = bb_u64(get_king_moves(Square::index(k.trailing_zeros() as usize)));
                s += self.w[791 + side] * ((ring & atk[opp]).count_ones() as f64) / 8.0;
            }
            // castling rights (raw)
            let color = if side == 0 { Color::White } else { Color::Black };
            let rights = board.castle_rights(color);
            if rights.short.is_some() {
                s += self.w[793 + 2 * side];
            }
            if rights.long.is_some() {
                s += self.w[794 + 2 * side];
            }
            // pawn structure
            let mut doubled = 0u32;
            let mut isolated = 0u32;
            for f in 0..8 {
                let cnt = (own_p & file_mask(f)).count_ones();
                if cnt > 1 {
                    doubled += cnt - 1;
                }
                if cnt > 0 && own_p & adj_files(f) == 0 {
                    isolated += cnt;
                }
            }
            let mut passed = 0u32;
            let mut m = own_p;
            while m != 0 {
                let sqi = m.trailing_zeros() as usize;
                if front_span(side == 0, sqi) & opp_p == 0 {
                    passed += 1;
                }
                m &= m - 1;
            }
            s += self.w[797 + side] * (doubled as f64) / 8.0;
            s += self.w[799 + side] * (isolated as f64) / 8.0;
            s += self.w[801 + side] * (passed as f64) / 8.0;
        }
        s
    }

    /// side-to-move tanh value (negamax convention)
    fn stm(&self, board: &Board) -> f64 {
        let v = self.score(board).tanh();
        if board.side_to_move() == Color::White {
            v
        } else {
            -v
        }
    }
}

const MATE: f64 = 10.0;

fn moves_of(board: &Board) -> Vec<Move> {
    let mut out = Vec::with_capacity(48);
    board.generate_moves(|ml| {
        out.extend(ml);
        false
    });
    out
}

fn is_capture(board: &Board, mv: Move) -> bool {
    bb_u64(board.colors(!board.side_to_move())) & (1u64 << mv.to as usize) != 0
        || (board.piece_on(mv.from) == Some(Piece::Pawn) && Some(mv.to) == board.en_passant().map(|f| {
            let r = if board.side_to_move() == Color::White { Rank::Sixth } else { Rank::Third };
            Square::new(f, r)
        }))
}

fn victim_value(board: &Board, mv: Move) -> f64 {
    board.piece_on(mv.to).map(|p| TYPE_VAL[p as usize]).unwrap_or(1.0)
}

/// standard-UCI text for a move (cozy uses king-takes-rook for castling)
fn uci(board: &Board, mv: Move) -> String {
    if board.piece_on(mv.from) == Some(Piece::King)
        && bb_u64(board.colors(board.side_to_move())) & (1u64 << mv.to as usize) != 0
    {
        let rank = mv.from.rank();
        let dst = if mv.to.file() > mv.from.file() { File::G } else { File::C };
        return format!("{}{}", mv.from, Square::new(dst, rank));
    }
    format!("{}", mv)
}

// --- v2 (spec :Depth-per-second:): transposition table, killers, history, null move -------
const TT_BITS: usize = 20;                            // 2^20 entries
const TT_SIZE: usize = 1 << TT_BITS;
const BOUND_EXACT: u8 = 0;
const BOUND_LOWER: u8 = 1;
const BOUND_UPPER: u8 = 2;
const MAX_PLY: usize = 64;

#[derive(Clone, Copy)]
struct TtEntry {
    key: u64,
    depth: i32,
    score: f64,
    bound: u8,
    mv: Option<Move>,
}

impl Default for TtEntry {
    fn default() -> Self {
        TtEntry { key: 0, depth: -1, score: 0.0, bound: BOUND_EXACT, mv: None }
    }
}

struct SearchCtx<'a> {
    eval: &'a Eval,
    nodes: u64,
    tt: &'a mut Vec<TtEntry>,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: Vec<u64>,                                // from*64+to
}

impl<'a> SearchCtx<'a> {
    fn qsearch(&mut self, board: &Board, mut alpha: f64, beta: f64) -> (f64, Board) {
        self.nodes += 1;
        let moves = moves_of(board);
        if moves.is_empty() {
            let sc = if board.checkers() != BitBoard::EMPTY { -MATE } else { 0.0 };
            return (sc, board.clone());
        }
        let stand = self.eval.stm(board);
        if stand >= beta {
            return (stand, board.clone());
        }
        let mut best = stand;
        let mut best_leaf = board.clone();
        if stand > alpha {
            alpha = stand;
        }
        let mut caps: Vec<Move> = moves.into_iter().filter(|&m| is_capture(board, m)).collect();
        caps.sort_by(|&a, &b| victim_value(board, b).partial_cmp(&victim_value(board, a)).unwrap());
        for mv in caps {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let (sc, leaf) = self.qsearch(&child, -beta, -alpha);
            let sc = -sc;
            if sc > best {
                best = sc;
                best_leaf = leaf;
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                break;
            }
        }
        (best, best_leaf)
    }

    /// returns (score_stm, best_move, leaf_board). v2: TT (cutoffs at NON-PV nodes only, so
    /// the PV leaf stays exact for training), TT-move/killer/history ordering, null-move R=2.
    fn negamax(
        &mut self,
        board: &Board,
        depth: u32,
        ply: usize,
        mut alpha: f64,
        beta: f64,
    ) -> (f64, Option<Move>, Board) {
        self.nodes += 1;
        let moves = moves_of(board);
        if moves.is_empty() {
            let sc = if board.checkers() != BitBoard::EMPTY {
                -(MATE + depth as f64)                      // deeper depth remaining = faster mate
            } else {
                0.0
            };
            return (sc, None, board.clone());
        }
        if depth == 0 {
            let (sc, leaf) = self.qsearch(board, alpha, beta);
            return (sc, None, leaf);
        }
        let is_pv = beta - alpha > 1e-6 * 2.0;             // null-window nodes are non-PV
        let key = board.hash();
        let slot = (key as usize) & (self.tt.len() - 1);   // mask by ACTUAL table size (per-game
                                                           // tables in play_games are smaller)
        let mut tt_mv: Option<Move> = None;
        {
            let e = &self.tt[slot];
            if e.key == key && e.depth >= 0 {
                tt_mv = e.mv;
                if !is_pv && e.depth >= depth as i32 && e.score.abs() < MATE - 1.0 {
                    match e.bound {
                        BOUND_EXACT => return (e.score, e.mv, board.clone()),
                        BOUND_LOWER if e.score >= beta => return (e.score, e.mv, board.clone()),
                        BOUND_UPPER if e.score <= alpha => return (e.score, e.mv, board.clone()),
                        _ => {}
                    }
                }
            }
        }
        // null-move pruning: not in check, not a pawn-only endgame, non-PV, enough depth
        let in_check = board.checkers() != BitBoard::EMPTY;
        if !is_pv && !in_check && depth >= 3 {
            let stm = board.side_to_move();
            let heavy = bb_u64(board.colors(stm))
                & !bb_u64(board.pieces(Piece::Pawn))
                & !bb_u64(board.pieces(Piece::King));
            if heavy != 0 {
                if let Some(nb) = board.null_move() {
                    let (sc, _, _) = self.negamax(&nb, depth - 3, ply + 1, -beta, -beta + 1e-6);
                    if -sc >= beta {
                        return (beta, None, board.clone());
                    }
                }
            }
        }
        // ordering: TT move, captures (MVV-LVA), killers, history-scored quiets
        let mut scored: Vec<(f64, Move)> = moves
            .into_iter()
            .map(|m| {
                let s = if Some(m) == tt_mv {
                    1e12
                } else if is_capture(board, m) {
                    1e9 + victim_value(board, m) * 100.0
                        - board.piece_on(m.from).map(|p| TYPE_VAL[p as usize]).unwrap_or(0.0)
                } else if ply < MAX_PLY && self.killers[ply].contains(&Some(m)) {
                    1e6
                } else {
                    self.history[(m.from as usize) * 64 + (m.to as usize)] as f64
                };
                (s, m)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let alpha0 = alpha;
        let mut best = f64::NEG_INFINITY;
        let mut best_mv = None;
        let mut best_leaf = board.clone();
        for (_, mv) in scored {
            let mut child = board.clone();
            child.play_unchecked(mv);
            let (sc, _, leaf) = self.negamax(&child, depth - 1, ply + 1, -beta, -alpha);
            let sc = -sc;
            if sc > best {
                best = sc;
                best_mv = Some(mv);
                best_leaf = leaf;
            }
            if sc > alpha {
                alpha = sc;
            }
            if alpha >= beta {
                if !is_capture(board, mv) && ply < MAX_PLY {
                    if self.killers[ply][0] != Some(mv) {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = Some(mv);
                    }
                    self.history[(mv.from as usize) * 64 + (mv.to as usize)] +=
                        (depth as u64) * (depth as u64);
                }
                break;
            }
        }
        if best.abs() < MATE - 1.0 {
            let bound = if best <= alpha0 {
                BOUND_UPPER
            } else if best >= beta {
                BOUND_LOWER
            } else {
                BOUND_EXACT
            };
            self.tt[slot] = TtEntry { key, depth: depth as i32, score: best, bound, mv: best_mv };
        }
        (best, best_mv, best_leaf)
    }
}

#[pyclass]
struct Searcher {
    eval: Eval,
    tt: Vec<TtEntry>,                                  // persists across calls (reuse within a game)
}

#[pymethods]
impl Searcher {
    #[new]
    fn new(weights: Vec<f64>, bias: f64) -> PyResult<Self> {
        if weights.len() != NFEAT {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} weights, got {}",
                NFEAT,
                weights.len()
            )));
        }
        let mut w = [0.0; NFEAT];
        w.copy_from_slice(&weights);
        Ok(Searcher { eval: Eval { w, bias }, tt: vec![TtEntry::default(); TT_SIZE] })
    }

    /// White-absolute pre-tanh score of a FEN (parity testing)
    fn score(&self, fen: &str) -> PyResult<f64> {
        let board: Board = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e:?}")))?;
        Ok(self.eval.score(&board))
    }

    /// (best_uci, white_value_tanh, leaf_fen, predicted_reply_uci, nodes)
    fn search(&mut self, fen: &str, depth: u32) -> PyResult<(String, f64, String, String, u64)> {
        let board: Board = fen
            .parse()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("bad fen: {e:?}")))?;
        let mut ctx = SearchCtx {
            eval: &self.eval,
            nodes: 0,
            tt: &mut self.tt,
            killers: [[None; 2]; MAX_PLY],
            history: vec![0u64; 64 * 64],
        };
        let mut result = None;
        for d in 1..=depth {
            result = Some(ctx.negamax(&board, d, 0, f64::NEG_INFINITY, f64::INFINITY));
        }
        let (sc, mv, leaf) = result.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("depth 0"))?;
        let mv = mv.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("no legal moves"))?;
        // predicted reply: the child's best move at depth-1
        let mut child = board.clone();
        child.play_unchecked(mv);
        let pred = if depth >= 2 {
            let (_, pm, _) = ctx.negamax(&child, depth - 1, 0, f64::NEG_INFINITY, f64::INFINITY);
            pm.map(|m| uci(&child, m)).unwrap_or_default()
        } else {
            String::new()
        };
        let white_val = if board.side_to_move() == Color::White { sc } else { -sc };
        Ok((
            uci(&board, mv),
            white_val.clamp(-1.0, 1.0),
            format!("{leaf}"),
            pred,
            ctx.nodes,
        ))
    }
}

// --- Merge 9 (spec/parallel-generation.spec.md): native parallel game generation ----------
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % (n as u64)) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

/// one generation game; returns (z_white, agent_white, per-agent-move records)
/// record = (leaf_fen, white_value, predicted: was the reply to the PREVIOUS agent move the
/// one its search expected). opp_depth 0 = uniform-random opponent.
#[allow(clippy::too_many_arguments)]
fn play_one(
    agent: &Eval,
    opp: &Eval,
    opp_depth: u32,
    depth: u32,
    eps: f64,
    ply_cap: u32,
    seed: u64,
    agent_white: bool,
) -> (f64, bool, Vec<(String, f64, bool)>) {
    let mut rng = XorShift(seed | 1);
    let mut tt_a = vec![TtEntry::default(); 1 << 16];
    let mut tt_o = vec![TtEntry::default(); 1 << 16];
    let mut board = Board::default();
    let mut recs: Vec<(String, f64, bool)> = Vec::with_capacity(64);
    let mut pending_pred: Option<Move> = None;
    let mut reply_actual: Option<Move> = None;
    let mut plies = 0u32;
    let z;
    loop {
        let moves = moves_of(&board);
        if moves.is_empty() {
            z = if board.checkers() != BitBoard::EMPTY {
                if board.side_to_move() == Color::White { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            break;
        }
        if plies >= ply_cap {
            z = 0.0;
            break;
        }
        let my_turn = (board.side_to_move() == Color::White) == agent_white;
        let mv;
        if my_turn {
            if eps > 0.0 && rng.unit() < eps {
                mv = moves[rng.below(moves.len())];         // ε: state-coverage exploration
                let predicted = pending_pred.is_some() && reply_actual == pending_pred;
                let _ = predicted;                          // ε-moves record nothing (no search value)
                pending_pred = None;
                reply_actual = None;
            } else {
                let mut ctx = SearchCtx {
                    eval: agent,
                    nodes: 0,
                    tt: &mut tt_a,
                    killers: [[None; 2]; MAX_PLY],
                    history: vec![0u64; 64 * 64],
                };
                let (sc, bm, leaf) = ctx.negamax(&board, depth, 0, f64::NEG_INFINITY, f64::INFINITY);
                let best = bm.unwrap_or(moves[0]);
                let predicted = pending_pred.is_some() && reply_actual == pending_pred;
                let white_val = if board.side_to_move() == Color::White { sc } else { -sc };
                recs.push((format!("{leaf}"), white_val.clamp(-1.0, 1.0), predicted));
                // predicted reply for the NEXT record: child's best at depth-1
                let mut child = board.clone();
                child.play_unchecked(best);
                pending_pred = if depth >= 2 {
                    let (_, pm, _) = ctx.negamax(&child, depth - 1, 0, f64::NEG_INFINITY, f64::INFINITY);
                    pm
                } else {
                    None
                };
                reply_actual = None;
                mv = best;
            }
        } else {
            if opp_depth == 0 {
                mv = moves[rng.below(moves.len())];
            } else {
                let mut ctx = SearchCtx {
                    eval: opp,
                    nodes: 0,
                    tt: &mut tt_o,
                    killers: [[None; 2]; MAX_PLY],
                    history: vec![0u64; 64 * 64],
                };
                let (_, bm, _) = ctx.negamax(&board, opp_depth, 0, f64::NEG_INFINITY, f64::INFINITY);
                mv = bm.unwrap_or(moves[0]);
            }
            if reply_actual.is_none() {
                reply_actual = Some(mv);
            }
        }
        board.play_unchecked(mv);
        plies += 1;
    }
    (z, agent_white, recs)
}

/// Parallel generation: n_games across `threads` OS threads. Opponent = (opp_weights, opp_depth);
/// opp_depth 0 ignores opp_weights (uniform random). Returns [(z, agent_white,
/// [(leaf_fen, white_value, predicted)])]. seed varies per game; colors alternate.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn play_games(
    py: Python<'_>,
    weights: Vec<f64>,
    bias: f64,
    opp_weights: Vec<f64>,
    opp_bias: f64,
    opp_depth: u32,
    n_games: usize,
    threads: usize,
    depth: u32,
    eps: f64,
    ply_cap: u32,
    seed: u64,
) -> PyResult<Vec<(f64, bool, Vec<(String, f64, bool)>)>> {
    if weights.len() != NFEAT || opp_weights.len() != NFEAT {
        return Err(pyo3::exceptions::PyValueError::new_err("weights must be 809-dim"));
    }
    let mut wa = [0.0; NFEAT];
    wa.copy_from_slice(&weights);
    let mut wo = [0.0; NFEAT];
    wo.copy_from_slice(&opp_weights);
    let agent = Eval { w: wa, bias };
    let opp = Eval { w: wo, bias: opp_bias };
    let results = py.allow_threads(|| {
        let agent = &agent;
        let opp = &opp;
        let nt = threads.max(1);
        std::thread::scope(|s| {
            let mut handles = Vec::new();
            for t in 0..nt {
                handles.push(s.spawn(move || {
                    let mut out = Vec::new();
                    let mut g = t;
                    while g < n_games {
                        out.push((
                            g,
                            play_one(agent, opp, opp_depth, depth, eps, ply_cap,
                                     seed.wrapping_add(g as u64).wrapping_mul(0x9E3779B97F4A7C15),
                                     g % 2 == 0),
                        ));
                        g += nt;
                    }
                    out
                }));
            }
            let mut all: Vec<(usize, (f64, bool, Vec<(String, f64, bool)>))> =
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
            all.sort_by_key(|x| x.0);
            all.into_iter().map(|x| x.1).collect::<Vec<_>>()
        })
    });
    Ok(results)
}

#[pymodule]
fn rsearch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Searcher>()?;
    m.add_function(wrap_pyfunction!(play_games, m)?)?;
    Ok(())
}
